"""
RouterClient — drive which model(s) the llama-server router keeps resident in VRAM.

For biology researchers: our offline engine may run several vision models (one to read pages, one
to locate tables, one to describe figures), but a limited-VRAM GPU cannot hold all of them at
once. The models live behind one llama-server running in "router" mode: you ask it to load or
unload a model by name, and it juggles GPU memory for you.

This module is the small, explicit remote control the pipeline uses to say "for this phase, keep
exactly these models resident." Which model names correspond to which role is decided entirely by
config (the strategy's profile_registry) and passed in as arguments — this module never names a
model itself. It exists because two facts make the naive approach unsafe:

  1. Some models are small enough to be co-resident (the reader + the table locator fit together),
     while a larger model may fit with NOTHING else. Loading such a model while others are resident
     would exceed the card, so before switching phases we must explicitly UNLOAD what we no longer
     need rather than trust the router's automatic eviction.
  2. `POST /models/load` returns as soon as the request is ACCEPTED, not when the weights are on
     the GPU. Timing that call reports a fraction of a second and is wrong. We must POLL
     `/v1/models` until the model reports status "loaded" before sending it any work.

The server is launched and owned OUTSIDE this process (the user starts it). This client only talks
to its HTTP API; it never spawns or kills the server.
"""

import logging
import time
from typing import List, Optional, Set

import requests


# The router reports each model's readiness under status.value; "loaded" means weights resident.
LOADED_STATE = "loaded"

# Poll cadence while waiting for a model to finish loading. Cold loads of the 7 GB OlmOCR from
# disk take ~12 s; a 2 s poll keeps the wait responsive without hammering the server.
POLL_INTERVAL_SECONDS = 2.0


class RouterClient:
    """
    A thin, poll-correct client for the llama-server model-management API.

    Usage (model names come from the caller's config, shown here as placeholders):
        router = RouterClient("http://localhost:9931")
        router.set_resident([reader_model, layout_model])   # Phase A: co-resident readers
        ... run extraction + layout calls ...
        router.set_resident([understanding_model])          # Phase B: one swap, exclusive
    """

    def __init__(self, base_url: str, logger: Optional[logging.Logger] = None):
        """
        Args:
            base_url: root URL of the running router, e.g. "http://localhost:9931" (no trailing
                path). The /models and /v1/models endpoints hang off this.
            logger: optional logger; a module logger is used if omitted.
        """
        self.base_url = base_url.rstrip("/")
        self.logger = logger or logging.getLogger("RouterClient")

    # -- introspection --------------------------------------------------------------------

    def loaded_models(self) -> Set[str]:
        """
        The set of model names currently reporting status "loaded".

        Returns:
            Model ids the router says are resident right now. Empty set if the server cannot be
            reached or returns nothing (callers treat that as "nothing resident").
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
        except requests.RequestException as error:
            self.logger.warning(f"Could not list router models: {error}")
            return set()

        resident = set()
        for entry in response.json().get("data", []):
            status = entry.get("status")
            # status may be {"value": "loaded"} on the router, or absent on plainer builds.
            state = status.get("value") if isinstance(status, dict) else status
            if state == LOADED_STATE:
                resident.add(entry.get("id"))
        return resident

    # -- primitives -----------------------------------------------------------------------

    def _wait_until_loaded(self, model: str, timeout: int) -> bool:
        """
        Block until `model` reports "loaded", or `timeout` seconds elapse.

        This is the step that makes loads correct: /models/load returns an async acknowledgement,
        so only the /v1/models status tells us the weights are actually resident.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if model in self.loaded_models():
                return True
            time.sleep(POLL_INTERVAL_SECONDS)
        return False

    def load(self, model: str, timeout: int = 180) -> bool:
        """
        Load `model` and wait until it is truly resident.

        Args:
            model: model id as defined in model.ini (e.g. "glm-ocr").
            timeout: seconds to wait for the weights to reach "loaded".

        Returns:
            True once the model is resident; False on request error or timeout.
        """
        if model in self.loaded_models():
            self.logger.debug(f"Model '{model}' already resident")
            return True

        self.logger.info(f"Loading model '{model}' ...")
        start = time.time()
        try:
            requests.post(f"{self.base_url}/models/load", json={"model": model}, timeout=timeout)
        except requests.RequestException as error:
            self.logger.error(f"Load request for '{model}' failed: {error}")
            return False

        if self._wait_until_loaded(model, timeout):
            self.logger.info(f"Model '{model}' resident in {time.time() - start:.1f}s")
            return True

        self.logger.error(f"Model '{model}' never reached '{LOADED_STATE}' within {timeout}s")
        return False

    def unload(self, model: str, timeout: int = 60) -> bool:
        """
        Unload `model`, freeing its VRAM. A model that is not loaded is a no-op success.
        """
        if model not in self.loaded_models():
            return True

        self.logger.info(f"Unloading model '{model}' ...")
        try:
            requests.post(f"{self.base_url}/models/unload", json={"model": model}, timeout=timeout)
        except requests.RequestException as error:
            self.logger.error(f"Unload request for '{model}' failed: {error}")
            return False
        return True

    # -- the workhorse --------------------------------------------------------------------

    def set_resident(self, models: List[str], timeout: int = 180) -> bool:
        """
        Make the resident set EXACTLY `models`: unload anything else, then load each of these.

        This is the one call each phase uses. Unloading first is load-bearing, not tidiness — it
        is what frees the VRAM the incoming models need (OlmOCR cannot fit beside GLM+DeepSeek).

        Args:
            models: the model ids that should be resident after this call (and only these).
            timeout: per-model load timeout.

        Returns:
            True only if every requested model ended up resident.
        """
        wanted = set(models)

        # 1. Evict every resident model the next phase does not need.
        for resident in self.loaded_models() - wanted:
            self.unload(resident)

        # 2. Load each wanted model, polling until the weights are actually on the GPU.
        all_ok = True
        for model in models:
            if not self.load(model, timeout=timeout):
                all_ok = False

        self.logger.info(f"Resident models now: {sorted(self.loaded_models())} (wanted {sorted(wanted)})")
        return all_ok
