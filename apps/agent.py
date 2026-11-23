import json
import re

import torch
from transformers import AutoTokenizer, TextStreamer

from apps.tools import TOOLS_FUNCTIONS, TOOLS_SIGNATURE
from model.model import MinimindCausalLLM
from model.model_config import MinimindConfig

class ColoredStreamer(TextStreamer):
    RESET = "\033[0m"
    COLORS = [
        (("<think>", "</think>"), "\033[90m"),
        (("<tool_call>", "</tool_call>"), "\033[90m"),
        (("<tool_response>", "</tool_response>"), "\033[90m")
    ]

    def on_finalized_text(self, text, stream_end=False):
        for (begin_tag, end_tag), color in self.COLORS:
            text = text.replace(begin_tag, color + begin_tag)
            text = text.replace(end_tag, end_tag + self.RESET)

        if stream_end:
            text += self.RESET

        super().on_finalized_text(text, stream_end=stream_end)

def parse_tool_calls(text: str) -> list[dict]:
    """Extract calls outside reasoning blocks."""
    text = re.sub(
        r"<think>.*?(?:</think>|$)",
        "",
        text,
        flags=re.DOTALL,
    )

    pattern = r"<tool_call>(.*?)</tool_call>"
    bodies = re.findall(pattern, text, flags=re.DOTALL)
    remainder = re.sub(pattern, "", text, flags=re.DOTALL)

    if "<tool_call>" in remainder or "</tool_call>" in remainder:
        raise ValueError("Incomplete <tool_call> block.")

    calls = []
    for body in bodies:
        call = json.loads(body)

        assert isinstance(call, dict)

        name = call.get("name")
        arguments = call.get("arguments", {})
        if isinstance(arguments, str): 
            arguments = json.loads(arguments)

        calls.append({"name": name, "arguments": arguments})
    return calls


class Agent:
    def __init__(
        self,
        model: MinimindCausalLLM,
        tokenizer: AutoTokenizer,
        device,
        enable_tools=False
    ):

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.enable_tools = enable_tools

        self.temperature = 0.99
        self.top_p = 0.95
        self.top_k = 20
        self.max_generate = 512
        self.max_tool_round = 4

        self.messages = [{"role": "system", "content": ""}]
        self.streamer = ColoredStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        :param logits: torch.Tensor, [1, vocab_size]
        :return: 
        """
        if self.temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits.float() / self.temperature


        if self.top_k > 0:
            top_values = torch.topk(logits, k=self.top_k, dim=-1).values
            cutoff = top_values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, -float("inf"))

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True) # [1, vocab_size], [1, vocab_size]
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) # [1, vocab_size]
            sorted_mask = cumulative_probs > self.top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            masked_indices = sorted_indices[0, sorted_mask[0]]
            logits[0, masked_indices] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1) # [1,1]
        return next_token

    @torch.inference_mode()
    def generate(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=TOOLS_SIGNATURE if self.enable_tools else None
        )

        prompt += "<|im_start|>assistant\n"
        # prompt += "<think>\n\n</think>\n\n"

        input_ids = self.tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        past_kv = [None] * self.model.config.num_hidden_layers
        generated_ids = []

        for _ in range(self.max_generate):
            past_length = 0 if past_kv[0] is None else past_kv[0][0].shape[1]
            position_idxs = torch.arange(
                past_length,
                past_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.device,
            )[None, :]

            logits, past_kv, _ = self.model(
                input_ids=input_ids,
                position_idxs=position_idxs,
                past_kv=past_kv,
                return_cache=True,
                logits_to_keep=1,
            )

            next_token = self._sample(logits[:, -1, :])
            next_token_id = next_token.item()

            if next_token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)
            self.streamer.put(next_token.cpu())
            input_ids = next_token

        self.streamer.end()
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        

    def run(self):
        while True:
            user_input = input(("=" * 50) + "\nYou: ").strip()

            messages = self.messages + [{"role": "user", "content": user_input}]

            for round_idx in range(self.max_tool_round+1):

                print("Assistant: ", end="", flush=True)
                response = self.generate(messages)
                messages.append({"role": "assistant", "content": response})

                # Process tool call
                if not self.enable_tools: break
                calls = parse_tool_calls(response)
                if len(calls) == 0: break
                if round_idx == self.max_tool_round: raise RuntimeError("Tool call: Max tool round")

                for call in calls:
                    name = call["name"]
                    arguments = call["arguments"]

                    if not isinstance(name, str) or name not in TOOLS_FUNCTIONS:
                        raise ValueError(f"Unknown tool: {name}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be an object.")

                    result = TOOLS_FUNCTIONS[name](**arguments)
                    result = json.dumps(result, ensure_ascii=False)
                    messages.append({
                        "role": "tool",
                        "content": result,
                    })

                    self.streamer.put(
                        self.tokenizer(
                            "<tool_response>\n" + result + "\n</tool_response>\n",
                            add_special_tokens=False,
                            truncation=False,
                            return_tensors="pt",
                        ).input_ids
                    )


            self.messages = messages

if __name__ == "__main__":
    device = torch.device("cuda:0")
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    
    # model
    config = MinimindConfig()
    model = MinimindCausalLLM(config).to(device)
    ckpt_path = "ckpts/sft.pth"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


    agent = Agent(model, tokenizer, device, enable_tools=True)
    agent.run()


