# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export pi0-FAST as programs that generate its action tokens, and a decoding the bundle runs without PyTorch.

pi0-FAST answers each frame with text: `Action: `, the action chunk's FAST tokens, then `|`. It generates the tokens
one at a time with a KV cache, and turns them into actions with the FAST tokenizer in Python, which a program cannot
run. One action chunk runs as:

::

    prefill       the frame -> the first token, and the prompt's padding mask and KV cache
    step x255     the tokens so far and the KV cache -> the next token, and the cache with its keys and values
    (decoding)    the tokens -> the normalized actions, in NumPy (`action_tokens` in the manifest)
    actions       the normalized actions -> the actions in robot units

A program's tensors keep one size, so the cache has room for every token from the start, and each step attends only
to the tokens generated before it. The policy stops generating once every answer has ended; a program cannot stop,
so the bundle always runs the `max_decoding_steps` steps, and the decoding cuts the answer at its end as the policy
does. The prompt writes the state's bin numbers as text, which the program writes in with tensor operations (see
[`StatePromptTokens`]), so the bundle runs only the task given at export.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.bundle.action_tokens import ActionDecoder
from lerobot.bundle.manifest import (
    ACTION_TOKENIZER,
    DECODED_ACTIONS,
    GENERATED_TOKENS,
    ActionTokens,
    ProgramRun,
)
from lerobot.export.recipe import (
    ActionChunkRecipe,
    ActionTokenData,
    Part,
    RunPlan,
    preprocessor_renames,
    random_frame,
    to_policy_input,
)
from lerobot.processor import (
    EnvTransition,
    PolicyProcessorPipeline,
    ProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import _transformers_available

from ..common.state_prompt import StatePromptTokens, extreme_states
from ..pi05.export_pi05 import PI05Actions
from ..utils import is_exporting
from .modeling_pi0_fast import PI0FastPolicy
from .processor_pi0_fast import Pi0FastPrepareStateAndLanguageTokenizerProcessorStep

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache

if TYPE_CHECKING:
    from lerobot.configs import PolicyFeature
    from lerobot.export.configs import ExportConfig
    from lerobot.policies import PreTrainedPolicy

    from .configuration_pi0_fast import PI0FastConfig

PREFILL_OUTPUTS = [GENERATED_TOKENS, "prefix_pad_mask", "prefix_cache", "generated_cache"]
STEP_INPUTS = [GENERATED_TOKENS, "step", "prefix_pad_mask", "prefix_cache", "generated_cache"]
STEP_OUTPUTS = [GENERATED_TOKENS, "generated_cache"]
LANGUAGE_MODEL = "model.paligemma_with_expert.paligemma.model.language_model"
# The modules a decoding step runs: not the vision tower.
STEP_MODULES = (LANGUAGE_MODEL, "model.paligemma_with_expert.paligemma.lm_head")
# Answers made of random FAST tokens that the bundle's decoding must turn into the policy's actions at export.
DECODING_CHECKS = 16


class _PolicyPart(nn.Module):
    """A part that runs some of the policy's own code.

    `torch.export` keeps every weight of the module it exports, used or not. So the policy is held as a plain
    reference, not a submodule, and only the modules the part runs are registered: the program holds their weights.
    """

    def __init__(self, policy: PI0FastPolicy, modules: tuple[str, ...]):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.weights = nn.ModuleList(policy.get_submodule(name) for name in modules)


class PI0FastPrefill(_PolicyPart):
    """The raw frame in; the first token, the prompt's padding mask and KV cache, and an empty cache for the rest out.

    The tokens come out as one `[max_decoding_steps]` tensor that the steps fill, the first one written. The KV cache
    of all layers comes out stacked, `[layers, keys and values, 1, heads, positions, width]`.
    """

    def __init__(
        self,
        policy: PI0FastPolicy,
        preprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        task: str,
        max_decoding_steps: int,
    ):
        super().__init__(policy, ("model",))
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.task = task
        self.max_decoding_steps = max_decoding_steps

    def forward(self, *frame: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        observation: dict[str, Any] = {
            name: to_policy_input(x, name in self.visual)
            for name, x in zip(self.frame_names, frame, strict=True)
        }
        observation["task"] = self.task
        observation["robot_type"] = ""
        batch = self.preprocessor(observation)
        policy = self.policy
        # The policy's own image resize and padding, which it exposes under no public name.
        images, img_masks = policy._preprocess_images(batch)
        token, cache, pad_mask = policy.model.prefill_fast(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        )
        keys_values = torch.stack([torch.stack((layer.keys, layer.values)) for layer in cache.layers])
        rest = self.max_decoding_steps - 1
        tokens = torch.cat((token[0], token.new_zeros(rest)))
        generated = keys_values.new_zeros((*keys_values.shape[:4], rest, keys_values.shape[5]))
        return tokens, pad_mask, keys_values, generated


class PI0FastStep(_PolicyPart):
    """One decoding step: the tokens so far, the step's index `t` and the KV cache in; the tokens with token `t`
    written, and the generated tokens' cache with the keys and values of token `t - 1` written, out.

    The cache holds the prompt, then one slot per generated token. Step `t` reads token `t - 1`, which the policy's
    `decode_step_fast` appends to the cache, and attends to the prompt, the tokens before it and itself.
    """

    def __init__(self, policy: PI0FastPolicy):
        super().__init__(policy, STEP_MODULES)

    def forward(
        self,
        tokens: Tensor,
        step: Tensor,
        prefix_pad_mask: Tensor,
        prefix_cache: Tensor,
        generated_cache: Tensor,
    ) -> tuple[Tensor, Tensor]:
        keys_values = torch.cat((prefix_cache, generated_cache), dim=4)
        cache = DynamicCache(tuple((layer[0], layer[1], None) for layer in keys_values.unbind(0)))
        # The slots of tokens not generated yet hold nothing, so they are left out of attention.
        written = torch.arange(generated_cache.shape[4], device=step.device) < step - 1
        pad_mask = torch.cat((prefix_pad_mask, written[None], torch.ones_like(prefix_pad_mask[:, :1])), dim=1)
        token, cache = self.policy.model.decode_step_fast(
            tokens.index_select(0, step - 1)[None], cache, pad_mask
        )
        last = torch.stack(
            [torch.stack((layer.keys[:, :, -1:], layer.values[:, :, -1:])) for layer in cache.layers]
        )
        return tokens.index_copy(0, step, token[0]), generated_cache.index_copy(4, step - 1, last)


class PI0FastExportRecipe(ActionChunkRecipe):
    """pi0-FAST as programs that generate its tokens and a decoding the bundle runs: see this module's documentation.

    The test case is the default recipe's, made with the whole policy as `lerobot-rollout` runs it. Before anything is
    exported, the chain must generate the policy's tokens exactly for two different frames, and the bundle's decoding
    must give the policy's actions for them and for answers of random FAST tokens. The decoding inverts the DCT with
    a NumPy matrix product where the policy uses SciPy, so the actions are held to the export's tolerance.
    """

    exact = False

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:
        """Refuse a policy that does not decode greedily with its KV cache in float32, then make the test case."""
        config = cast("PI0FastConfig", policy.config)
        if config.temperature != 0:
            raise ValueError(
                f"This pi0-FAST checkpoint samples its tokens at temperature {config.temperature}, and the recipe "
                "exports greedy decoding only: set --policy.temperature=0."
            )
        if not config.use_kv_cache:
            raise ValueError(
                "The pi0-FAST recipe exports decoding with a KV cache, as the policy decodes by default: set "
                "--policy.use_kv_cache=true."
            )
        if config.dtype == torch.bfloat16:
            raise ValueError(
                "In bfloat16 the exported steps round differently from the policy and soon pick other tokens, so "
                "the pi0-FAST recipe exports float32 only: set --policy.dtype=float32."
            )
        super().__init__(policy, preprocessor, postprocessor, cfg)

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its state prompt and tokenizer steps replaced by [`_StatePromptStep`]."""
        (prompt,) = (
            step
            for step in self.preprocessor.steps
            if isinstance(step, Pi0FastPrepareStateAndLanguageTokenizerProcessorStep)
        )
        (tokenizer,) = (step for step in self.preprocessor.steps if isinstance(step, TokenizerProcessorStep))
        replaced = _StatePromptStep(prompt, tokenizer)
        steps = [
            replaced if step is tokenizer else step for step in self.preprocessor.steps if step is not prompt
        ]
        return replace(self.preprocessor, steps=steps)

    def parts(self) -> Iterator[Part]:
        """The prefill, the step and the actions, after checking the chain and the decoding against the policy."""
        policy = cast(PI0FastPolicy, self.policy)
        config = cast("PI0FastConfig", policy.config)
        frame_names = list(self._features)
        self._max_decoding_steps = config.max_decoding_steps
        prefill = PI0FastPrefill(
            policy,
            self.program_preprocessor(),
            frame_names,
            self._visual,
            self.task or "",
            self._max_decoding_steps,
        )
        step = PI0FastStep(policy)
        actions = PI05Actions(
            self.postprocessor,
            self._steps,
            cast("dict[str, PolicyFeature]", config.output_features)[ACTION].shape[0],
        )
        renames = preprocessor_renames(self.preprocessor)
        state_name = next(name for name in self._features if renames.get(name, name) == OBS_STATE)
        # A program takes only the inputs it reads: the state, for a policy trained on actions relative to it.
        self._state_names = [] if actions.mask is None else [state_name]
        self._action_tokens = self._decoding(policy)
        decoder = ActionDecoder(json.loads(self._action_tokens.tokenizer), self._action_tokens.spec)
        self._check_decoding(decoder)
        chain = (prefill, step, decoder, actions)

        frame = random_frame(self._features, self._generator)
        decoded = []
        for case_frame, expected in (
            (self._case.frame, self._case.actions),
            (frame, self._reference(frame, None)),
        ):
            tokens, frame_decoded, program_actions = self._run(chain, case_frame)
            self._check_tokens(tokens, case_frame)
            self._check_program(program_actions, expected)
            decoded.append(frame_decoded)

        inputs = self._inputs(self._case.frame, None)
        with torch.no_grad():
            tokens_0, pad_mask, prefix_cache, generated = prefill(*inputs)
        yield Part("prefill", prefill, inputs, frame_names, PREFILL_OUTPUTS)
        step_inputs = (tokens_0, torch.tensor([1], device=self._device), pad_mask, prefix_cache, generated)
        yield Part("step", step, step_inputs, STEP_INPUTS, STEP_OUTPUTS)
        del tokens_0, pad_mask, prefix_cache, generated, step_inputs
        actions_inputs = self._actions_inputs(decoded[0], self._case.frame)
        yield Part("actions", actions, actions_inputs, [DECODED_ACTIONS, *self._state_names], [ACTION])

    def plan(self) -> RunPlan:
        """The prefill, the step once per token, the actions, the step indices, and the decoding."""
        steps = range(1, self._max_decoding_steps)
        runs = [
            ProgramRun("prefill", "prefill", list(self._features), PREFILL_OUTPUTS),
            *(
                ProgramRun(
                    f"step_{t}", "step", [GENERATED_TOKENS, f"step_{t}", *STEP_INPUTS[2:]], STEP_OUTPUTS
                )
                for t in steps
            ),
            ProgramRun("actions", "actions", [DECODED_ACTIONS, *self._state_names], [ACTION]),
        ]
        return RunPlan(
            runs=runs,
            constants={f"step_{t}": np.array([t], np.int64) for t in steps},
            action_tokens=self._action_tokens,
        )

    def _decoding(self, policy: PI0FastPolicy) -> ActionTokenData:
        """The FAST tokenizer and how the policy reads its answer, as `detokenize_actions` does."""
        config = cast("PI0FastConfig", policy.config)
        # The tokenizer the policy decodes with, which it keeps under no public name.
        text = policy._paligemma_tokenizer
        fast = policy.action_tokenizer
        spec = ActionTokens(
            tokenizer=ACTION_TOKENIZER,
            clean_up_spaces=bool(fast.bpe_tokenizer.clean_up_tokenization_spaces),
            id_offset=text.vocab_size - 1 - config.fast_skip_tokens,
            end=text.convert_tokens_to_ids("|"),
            prefix=text.encode("Action: ", add_special_tokens=False),
            start=text.convert_tokens_to_ids(["Action", ":"]) if config.validate_action_token_prefix else [],
            min_token=int(fast.min_token),
            scale=float(fast.scale),
            horizon=config.n_action_steps,
            action_dim=cast("dict[str, PolicyFeature]", config.output_features)[ACTION].shape[0],
        )
        return ActionTokenData(fast.bpe_tokenizer.backend_tokenizer.to_str(), spec)

    def _check_decoding(self, decoder: ActionDecoder) -> None:
        """Refuse a decoding that differs from the policy's `detokenize_actions` on answers of random FAST tokens.

        The answers cross the end marker and hold the answer's prefix again in the middle, and some hold ids the FAST
        tokenizer does not know or that it turns down, so every branch of the decoding is compared.
        """
        policy = cast(PI0FastPolicy, self.policy)
        spec = self._action_tokens.spec
        vocabulary = len(policy.action_tokenizer.bpe_tokenizer)
        for i in range(DECODING_CHECKS):
            fast_ids = torch.randint(vocabulary, (40,), generator=self._generator)
            if i % 4 == 1:
                fast_ids[5] = vocabulary + 3
            if i % 4 == 2:
                fast_ids[7] = -1
            middle = torch.tensor(spec.prefix if i % 2 else [spec.end])
            first, second = spec.id_offset - fast_ids[:20], spec.id_offset - fast_ids[20:]
            answer = torch.cat((torch.tensor(spec.prefix), first, middle, second))
            expected = policy.detokenize_actions(answer, spec.horizon, spec.action_dim).numpy()
            difference = float(np.abs(decoder(answer.numpy()) - expected).max())
            if not difference <= self.tolerance:
                raise ValueError(
                    f"The bundle's decoding of pi0-FAST's tokens differs from the policy's by {difference:.3g} on "
                    "an answer of random FAST tokens, so this checkpoint's tokenizer cannot be decoded this way."
                )

    def _check_tokens(self, tokens: list[int], frame: dict[str, np.ndarray]) -> None:
        """Refuse a chain that generates other tokens than the policy for `frame`, up to its answer's end."""
        policy = cast(PI0FastPolicy, self.policy)
        self._reset()
        with torch.no_grad():
            batch = self.preprocessor(self._observation(frame, self.task))
            images, img_masks = policy._preprocess_images(batch)
            expected = policy.model.sample_actions_fast_kv_cache(
                images,
                img_masks,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                max_decoding_steps=self._max_decoding_steps,
            )[0].tolist()
        end = self._action_tokens.spec.end
        # The policy stops once its answer ends and leaves zeros after it.
        length = expected.index(end) + 1 if end in expected else len(expected)
        if tokens[:length] != expected[:length]:
            first = next(i for i, (a, b) in enumerate(zip(tokens, expected, strict=False)) if a != b)
            raise ValueError(
                f"The pi0-FAST chain generates token {tokens[first]} where the policy generates {expected[first]}, "
                f"at position {first} of {length}, so it cannot be exported this way."
            )

    def _run(
        self,
        chain: tuple[PI0FastPrefill, PI0FastStep, ActionDecoder, PI05Actions],
        frame: dict[str, np.ndarray],
    ) -> tuple[list[int], np.ndarray, np.ndarray]:
        """The chain in PyTorch for one frame: its tokens, the decoded actions, and the actions in robot units."""
        prefill, step, decoder, actions = chain
        with torch.no_grad():
            tokens, pad_mask, prefix_cache, generated = prefill(*self._inputs(frame, None))
            for t in range(1, self._max_decoding_steps):
                index = torch.tensor([t], device=self._device)
                tokens, generated = step(tokens, index, pad_mask, prefix_cache, generated)
            decoded = decoder(tokens.cpu().numpy())
            program_actions = actions(*self._actions_inputs(decoded, frame)).cpu().numpy()
        return tokens.tolist(), decoded, program_actions

    def _actions_inputs(self, decoded: np.ndarray, frame: dict[str, np.ndarray]) -> tuple[Tensor, ...]:
        return tuple(
            torch.from_numpy(x).to(self._device)
            for x in (decoded, *(frame[name] for name in self._state_names))
        )


class _StatePromptStep(ProcessorStep):
    """pi0-FAST's state prompt and tokenizer steps for one task, as tensor operations.

    The prompt's tokens before and after the state are the saved steps' on the first eager frame. Every eager frame is
    checked against the saved steps, and so are states beyond the first and last bins on the first one.
    """

    def __init__(
        self, prompt: Pi0FastPrepareStateAndLanguageTokenizerProcessorStep, tokenizer: TokenizerProcessorStep
    ) -> None:
        self.saved = (prompt, tokenizer)
        text = tokenizer.input_tokenizer
        self.pad_id = text.pad_token_id
        # The bin numbers -1 to 255 of the prompt's state, each with the space before it.
        self.pieces = [text.encode(f" {value}", add_special_tokens=False) for value in range(-1, 256)]
        self.prompt: StatePromptTokens | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not is_exporting():
            if self.prompt is None:
                self._fit(transition)
                for extreme in extreme_states(transition):
                    self._check(extreme)
            self._check(transition)
        return self._apply(transition)

    def _bins(self, state: Tensor) -> Tensor:
        """The state's bin numbers plus one, as the saved step digitizes them against 256 equal bins of [-1, 1]."""
        edges = -1 + torch.arange(256, dtype=torch.float32, device=state.device) / 128
        return (state.flatten()[:, None] >= edges).sum(-1)

    def _saved(self, transition: EnvTransition) -> dict[str, Any]:
        prompt, tokenizer = self.saved
        return tokenizer(prompt(transition))[TransitionKey.OBSERVATION] or {}

    def _apply(self, transition: EnvTransition) -> EnvTransition:
        if self.prompt is None:
            raise ValueError("The prompt is made from an eager call before the program is traced.")
        observation = dict(transition[TransitionKey.OBSERVATION] or {})
        ids, mask = self.prompt(self._bins(observation[OBS_STATE]))
        observation[OBS_LANGUAGE_TOKENS] = ids[None]
        observation[OBS_LANGUAGE_ATTENTION_MASK] = mask[None]
        return {**transition, TransitionKey.OBSERVATION: observation}

    def _fit(self, transition: EnvTransition) -> None:
        """Take the prompt around the state from the saved steps on this frame."""
        saved = self._saved(transition)
        ids, mask = saved[OBS_LANGUAGE_TOKENS][0], saved[OBS_LANGUAGE_ATTENTION_MASK][0]
        bins = self._bins((transition[TransitionKey.OBSERVATION] or {})[OBS_STATE])
        self.prompt = StatePromptTokens.around(
            ids[mask].tolist(), bins.tolist(), self.pieces, self.pad_id, len(ids), bins.device
        )

    def _check(self, transition: EnvTransition) -> None:
        """Refuse a prompt other than the saved steps' for this frame."""
        program = self._apply(transition)[TransitionKey.OBSERVATION] or {}
        saved = self._saved(transition)
        if not all(
            torch.equal(program[key], saved[key].to(program[key].device))
            for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
        ):
            state = (transition[TransitionKey.OBSERVATION] or {})[OBS_STATE].tolist()
            raise ValueError(
                "The program writes another prompt than pi0-FAST's state prompt and tokenizer steps for the "
                f"normalized state {state}: this tokenizer splits the state's numbers differently inside the "
                "prompt, so pi0-FAST cannot be exported this way."
            )

    def transform_features(self, features):
        return features
