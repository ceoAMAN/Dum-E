# pyre-unsafe
"""Entry point for Sturnus."""
from __future__ import annotations

import argparse
import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import configs
from experts import ExpertPool
from gating import GateRouter


@dataclass
class BackgroundTask:
    thread: threading.Thread
    cancel_event: threading.Event


class SturnusEngine:
    def __init__(self) -> None:
        self.expert_pool = ExpertPool()
        self.router = GateRouter(self.expert_pool)
        self.history: List[str] = []
        self.background: Optional[BackgroundTask] = None

    def _context(self) -> str:
        return "\n".join(self.history[-20:])

    def _cancel_background(self) -> None:
        if self.background is None:
            return
        self.background.cancel_event.set()
        self.background.thread.join(timeout=0.1)
        self.background = None

    def _start_background(self, text: str, context: str, expert_indices: List[int], x_concurrency: int) -> None:
        cancel_event = threading.Event()

        def _worker() -> None:
            start = time.time()
            while time.time() - start < configs.DEAD_TIME_MIN_SECS:
                if cancel_event.is_set():
                    return
                time.sleep(0.05)
            if cancel_event.is_set():
                return

            async def _run():
                await asyncio.wait_for(
                    self.router.run_timeline_b(
                        text,
                        context,
                        expert_indices,
                        mode=1,
                        x_concurrency=x_concurrency,
                    ),
                    timeout=configs.TIMELINE_B_BUDGET_SECS,
                )

            try:
                asyncio.run(_run())
            except Exception:
                return

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        self.background = BackgroundTask(thread=thread, cancel_event=cancel_event)

    async def process_input(self, text: str) -> Dict[str, Any]:
        context = self._context()
        decision = self.router.route(text, context)
        if decision.timeline == "A":
            result = await self.router.run_timeline_a(text)
            self._cancel_background()
            self._start_background(text, context, decision.expert_indices, decision.x)
        else:
            result = await self.router.run_timeline_b(
                text,
                context,
                decision.expert_indices,
                mode=2,
                x_concurrency=decision.x,
            )
        self.history.append(text)
        return {
            "decision": decision,
            "result": result,
        }


async def _run_once(prompt: str) -> None:
    engine = SturnusEngine()
    output = await engine.process_input(prompt)
    print("[main] decision:", output["decision"])
    print("[main] output:", output["result"]["output_text"])


def self_test() -> None:
    print("[main] self-test")
    asyncio.run(_run_once("Test input for Sturnus"))


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run the Sturnus engine")
    parser.add_argument("--prompt", type=str, help="Single prompt to run")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat loop")
    parser.add_argument("--max-turns", type=int, default=0, help="Stop after N turns in interactive mode")
    parser.add_argument("--json", action="store_true", help="Print raw result dict")
    args = parser.parse_args()

    engine = SturnusEngine()

    async def _run_prompt(text: str) -> None:
        result = await engine.process_input(text)
        if args.json:
            print(result)
        else:
            print(result["result"]["output_text"])

    if args.prompt:
        asyncio.run(_run_prompt(args.prompt))
        return

    if args.interactive:
        turns = 0
        try:
            while True:
                if args.max_turns and turns >= args.max_turns:
                    break
                user_in = input("Sturnus> ").strip()
                if not user_in:
                    break
                asyncio.run(_run_prompt(user_in))
                turns += 1
        finally:
            engine._cancel_background()
        return

    self_test()


if __name__ == "__main__":
    run_cli()
