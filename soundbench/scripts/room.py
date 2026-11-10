"""The room: players who listen to each other.

A piece opts in with a `[room]` table in its sidecar. Each player is one or more
MIDI voice tracks played by one pair of hands. The room replaces the generic
onset jitter of those tracks with three coupled layers:

Time (coupled timekeepers). Each player is a Wing-Kristofferson timekeeper with
motor noise, corrected toward the other players by linear phase correction
(Vorberg & Wing 1996; Repp 2005; Wing, Endo, Bradbury & Vorberg 2014, string
quartets). Simulated beat by beat, in order, with causal beat and completed-stroke
pulse cues:

    x_i(n+1) = x_i(n) + T_i(n) - T(n) - sum_j alpha_ij * A_ij(n)
    onset_i(n) = S(n) + x_i(n) + M_i(n)
    T_i(n) = T(n) * (1 + p_i(n)) + N(0, timekeeper * T(n))

T(n) is the score's expressive beat length (tempo map, rit, fermata, phrase
arches) and S(n) its running sum: the shared target. x_i is player i's timekeeper
phase relative to that target, M_i its motor noise, A_ij(n) = onset_i(n) -
onset_j(n) its asynchrony from a player j heard on beat n. p_i is i's period
deviation: a private, bounded random walk that relaxes slowly back to the score
and, by period correction beta, toward the ensemble's recent tempo. A player who
rests still listens (A is then taken against its silent internal beat); a player
who has no recent strike is not heard. Notes between beats sit on the player's own
beat, interpolated, plus their own motor noise and the player's feel.
At 0.8+, A uses the expected performed attack difference minus the intended feel
difference. Groove, chord bounds, strum, melodic anticipation and calibrated
instrument attack are projected by the performer. Offbeat strokes reveal a pulse
on the next beat; no future note is a correction source. A slow, authored-coupling
weighted mean-error integrator corrects follower periods toward the lead.

Tightness is a composed parameter, default 0.8, with inclusive-bar passage
overrides. At 0.5 the base parameters and random streams are byte-exact.
Innovation (timekeeper noise, period-walk noise and bound) scales as 2**(2-4*t).
Motor noise scales more gently, by 0.75 + 0.25*innovation. Phase and period
correction scale by 1 + 0.6*(t-0.5); strengthened phase gains have a total budget
of 0.85 per listener to avoid overcorrection. At 0.8+, relative feel span is bounded
to 8 ms, contracting around the ensemble's common mean rather than erasing its
common feel. Mean correction, offbeat cue strength and feel compression fade from
zero at 0.5 to full strength at 0.8. Expressive tempo, strum widths, melodic gestures,
velocity and loudness contagion keep their mechanisms. State carries across passages;
there is no reset or entrance snap. The endpoints are musical descriptions,
not fitted estimates of years of rehearsal.

Measured references and calibration are in references/room-tightness/REPORT.md.
Professional ensembles do not have one universal precision threshold: Malian
jembe trios measure 17 ms pairwise RMS, while Rasch's chamber ensembles measure
30-50 ms (as summarized by Jacoby, Polak & London 2021). The default targets the
lower end for this string band's accidental variation while retaining feel.

Loudness. Each player's swell (a two-beat average of its notes' intended velocity
against its own sixteen-beat level) spreads to the players who listen to it, a
beat later: L_i(n) = D_i(n) + C_i(n), C_i(n) = memory * C_i(n-1) + gamma *
sum_j w_ij (L_j(n-1) - L_i(n-1)), w the coupling matrix normalised per listener,
C capped at `contagion_cap`. The lead's line is its `voice` tracks. Under the
line's active notes (struck within the last beat), accompaniment within
`yield_semitones` steps back `yield` velocity units. A gap opens a beat after the
line's last note when its next note is at least `gap_beats` further on (a held
note or a rest); there accompaniment onsets step forward by `answer`, tapering to
nothing over two beats.

Feel. A per-player (or per-track) microtiming and accent profile on a grid that
restarts each bar. Each night redraws the profile's offsets around its mean and
scales its depth, within bounds; within a night each position wanders slowly from
bar to bar.

Sidecar:

  [room]
  lead = "piano"            one lead throughout, or [[room.lead]] passages below;
                            outside every passage the first player leads
  tightness = 0.8           0 = first rehearsal; 1 = long-established band; 0.5 = compatibility
  feel_span_ms = 8          permitted relative feel span at 0.8+; explicit wider passages allowed
  follow = 0.3              alpha of everyone toward the lead
  lead_listens = 0.07       alpha of the lead toward each other player
  listen = 0.07             alpha between two players neither of whom leads
  beta = 0.05               period correction toward the ensemble's recent tempo
  motor_ms = 5              motor noise SD
  timekeeper = 0.01         timekeeper SD as a fraction of the beat
  drift = 0.002             per-beat step SD of the private period walk
  drift_return = 0.02       per-beat relaxation of that walk toward the score
  drift_bound = 0.03        |period deviation| never exceeds this
  contagion = 0.3           gamma; contagion_memory = 0.4; contagion_cap = 6
  yield = 4                 yield_semitones = 7
  answer = 5                gap_beats = 1

  [room.players]            player = track selector or list of selectors
  piano = ["upper", "alto", "lower"]

  machine = "drums"         players that are clocks (a drum machine): exactly on the score's
                            time, no motor noise, drift or feel, listening to no one; make one
                            the lead and the others follow its grid

  [room.feel]               player or track = built-in profile name or inline table
  banjo = "scruggs"         (a track's feel overrides its player's)

  [[room.lead]]             the lead by passage (inclusive bars)
  bars = [21, 36]
  player = "banjo"
  voice = "banjolead"       optional: the tracks that carry its line (default: all)

  [[room.passage]]          composed tightness by inclusive bars; last matching entry wins
  bars = [21, 36]
  tightness = 0.95
  feel_span_ms = 25         optional: explicitly spacious relative feel

  [tracks]                 calibrated instrument attack reference, used by the room
  upper = {attack_latency_ms = 10}

  [[room.pair]]             coupling overrides, optionally for a passage
  players = ["bass", "guitar"]
  alpha = 0.3               or [a_listens_to_b, b_listens_to_a]
  bars = [1, 80]            optional

A feel table: subdivision (8 or 16), timing_ms and velocity (cycled from each bar
line), night_ms / night_velocity (SD of the per-night redraw, bounded at twice
that), night_scale (the depth varies by up to this fraction per night),
wander_ms / wander_velocity (SD of the slow within-night wander).
"""
from __future__ import annotations

import bisect
import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from soundbench.scripts.instrumentation import matching_tracks

DEFAULTS: dict[str, float] = {
    "tightness": 0.8, "feel_span_ms": 8.0,
    "follow": 0.3, "lead_listens": 0.07, "listen": 0.07, "beta": 0.05,
    "motor_ms": 5.0, "timekeeper": 0.01,
    "drift": 0.002, "drift_return": 0.02, "drift_bound": 0.03, "tempo_memory": 4.0,
    "contagion": 0.3, "contagion_memory": 0.4, "contagion_cap": 6.0,
    "yield": 4.0, "yield_semitones": 7.0, "answer": 5.0, "gap_beats": 1.0,
}
FEEL_KEYS = {"subdivision", "timing_ms", "velocity", "night_ms", "night_velocity",
             "night_scale", "wander_ms", "wander_velocity"}
FEELS: dict[str, dict[str, Any]] = {
    "straight": {"subdivision": 8, "timing_ms": [0], "velocity": [0],
                 "night_ms": 1.5, "night_velocity": 1, "night_scale": 0,
                 "wander_ms": 1.5, "wander_velocity": 1},
    # The forward roll T I M | T I M | T M on sixteenths, twice a bar: the thumb
    # early and accented, the middle finger ghosted and late, the last pair
    # pressing into the next downbeat.
    "scruggs": {"subdivision": 16,
                "timing_ms": [-3, 3, 9, -7, 2, 8, -10, -4],
                "velocity": [7, -1, -8, 5, -2, -8, 4, -6],
                "night_ms": 2.5, "night_velocity": 1.5, "night_scale": 0.25,
                "wander_ms": 2.5, "wander_velocity": 1.5},
    # Boom on the beat; the chuck laid back and a touch lighter.
    "boomchuck": {"subdivision": 8,
                  "timing_ms": [0, 6, 16, 6],
                  "velocity": [3, -2, -2, -3],
                  "night_ms": 2.5, "night_velocity": 1, "night_scale": 0.25,
                  "wander_ms": 2, "wander_velocity": 1},
    # The bass sits slightly ahead, the downbeat furthest.
    "bass-push": {"subdivision": 8,
                  "timing_ms": [-10, -6, -8, -6],
                  "velocity": [3, 0, 1, 0],
                  "night_ms": 2.5, "night_velocity": 1, "night_scale": 0.25,
                  "wander_ms": 2, "wander_velocity": 1},
}
WANDER_CORRELATION = 0.9  # bar to bar: a correlation time of about ten bars


def night_seed(base: int, night: int | None) -> int:
    """The seed of one night of a piece: the base seed itself when no night is given."""
    if night is None:
        return int(base)
    digest = hashlib.sha256(f"folk-variations night {int(base)} {int(night)}".encode()).digest()
    return int.from_bytes(digest[:6], "big")


def _stream(seed: int, name: str) -> random.Random:
    return random.Random(f"{seed}:{name}")


def _bars(value, what: str) -> tuple[int, int]:
    if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value)
            and 1 <= value[0] <= value[1]):
        raise ValueError(f"{what} bars must be [first, last] with 1 <= first <= last")
    return value[0], value[1]


@dataclass
class Room:
    players: dict[str, list[int]]
    player_of: dict[int, str]
    lead: str
    leads: list[tuple[int, int, str, list[int]]]  # first, last, player, voice tracks
    pairs: list[tuple[str, str, float, int, int]]  # listener, source, alpha, first, last
    params: dict[str, float]
    feels: dict[int, tuple[str, dict[str, Any]]]  # track -> (instance key, profile)
    order: list[str] = field(default_factory=list)
    passages: list[tuple[int, int, float]] = field(default_factory=list)
    feel_spans: list[tuple[int, int, float]] = field(default_factory=list)
    machines: frozenset[str] = frozenset()   # players that are clocks: exact, never listening

    def mean_weight(self, bar: int) -> float:
        return min(1.0, max(0.0, (self.tightness_at(bar)-0.5)/0.3))

    def feel_span_at(self, bar: int) -> float:
        for first, last, value in reversed(self.feel_spans):
            if first <= bar <= last:
                return value
        return self.params['feel_span_ms']

    def tightness_at(self, bar: int) -> float:
        for first, last, value in reversed(self.passages):
            if first <= bar <= last:
                return value
        return self.params["tightness"]

    def timing_factors(self, bar: int) -> tuple[float, float, float]:
        """Innovation, motor and listening factors; 0.5 is byte-exact compatibility."""
        value = self.tightness_at(bar)
        innovation = 2.0 ** (2.0 - 4.0 * value)
        return innovation, 0.75 + 0.25 * innovation, 1.0 + 0.6 * (value - 0.5)

    def lead_at(self, bar: int) -> str:
        for first, last, player, _ in reversed(self.leads):
            if first <= bar <= last:
                return player
        return self.lead

    def voice_at(self, bar: int) -> list[int]:
        """The tracks that carry the lead's line in this bar."""
        for first, last, player, voice in reversed(self.leads):
            if first <= bar <= last:
                return voice or self.players[player]
        return self.players[self.lead]

    def alpha(self, listener: str, source: str, bar: int) -> float:
        if listener in self.machines:
            return 0.0
        for who, whom, value, first, last in reversed(self.pairs):
            if who == listener and whom == source and first <= bar <= last:
                return value
        lead = self.lead_at(bar)
        if source == lead:
            return self.params["follow"]
        if listener == lead:
            return self.params["lead_listens"]
        return self.params["listen"]


def parse(settings: dict[str, Any], note_tracks: list[int], names: dict[int, str]) -> Room | None:
    table = settings.get("room")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError("room must be a table")
    unknown = set(table) - set(DEFAULTS) - {"lead", "players", "feel", "pair", "passage", "machine"}
    if unknown:
        raise ValueError(f"unknown room settings: {sorted(unknown)}")
    params = {key: float(table.get(key, value)) for key, value in DEFAULTS.items()}
    if any(not math.isfinite(v) or v < 0 for v in params.values()):
        raise ValueError("room parameters must be finite and nonnegative")
    if params["tempo_memory"] < 1:
        raise ValueError("room.tempo_memory must be at least one beat")

    if not 0 <= params["tightness"] <= 1:
        raise ValueError("room.tightness must be between 0 and 1")
    passages, feel_spans = [], []
    for entry in table.get("passage", []):
        if (not isinstance(entry, dict) or 'bars' not in entry
                or not (set(entry) & {'tightness', 'feel_span_ms'})
                or set(entry) - {'bars', 'tightness', 'feel_span_ms'}):
            raise ValueError("each [[room.passage]] needs bars and tightness or feel_span_ms")
        first, last = _bars(entry["bars"], "room.passage")
        if 'tightness' in entry:
            value = float(entry["tightness"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("room.passage tightness must be between 0 and 1")
            passages.append((first, last, value))
        if 'feel_span_ms' in entry:
            value = float(entry['feel_span_ms'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('room.passage feel_span_ms must be finite and nonnegative')
            feel_spans.append((first, last, value))

    def resolve(selector) -> list[int]:
        matches = matching_tracks(selector, note_tracks, names)
        if not matches:
            raise ValueError(f"room selector {selector!r} matched no note tracks")
        return matches

    players: dict[str, list[int]] = {}
    player_of: dict[int, str] = {}
    raw_players = table.get("players")
    if not isinstance(raw_players, dict) or not raw_players:
        raise ValueError("room needs a [room.players] table of player = [tracks]")
    for player, selectors in raw_players.items():
        selectors = selectors if isinstance(selectors, list) else [selectors]
        tracks = []
        for selector in selectors:
            for index in resolve(selector):
                if index in player_of:
                    raise ValueError(f"track {names.get(index)!r} belongs to two room players")
                player_of[index] = player
                tracks.append(index)
        players[player] = tracks

    def player(name, what) -> str:
        if name not in players:
            raise ValueError(f"{what} {name!r} is not a room player; players: {list(players)}")
        return name

    raw_lead = table.get("lead", [])
    lead = player(raw_lead, "room.lead") if isinstance(raw_lead, str) else next(iter(players))
    leads = []
    for entry in raw_lead if isinstance(raw_lead, list) else []:
        if not isinstance(entry, dict) or not {"bars", "player"} <= set(entry) \
                or set(entry) - {"bars", "player", "voice"}:
            raise ValueError("each [[room.lead]] needs bars = [first, last], player and optional voice")
        first, last = _bars(entry["bars"], "room.lead")
        who = player(entry["player"], "room.lead player")
        voice = entry.get("voice", [])
        voice = [t for selector in (voice if isinstance(voice, list) else [voice]) for t in resolve(selector)]
        if any(player_of.get(t) != who for t in voice):
            raise ValueError(f"room.lead voice must be tracks of {who!r}")
        leads.append((first, last, who, voice))

    pairs = []
    for entry in table.get("pair", []):
        if not isinstance(entry, dict) or not {"players", "alpha"} <= set(entry) \
                or set(entry) - {"players", "alpha", "bars"}:
            raise ValueError("each [[room.pair]] needs players = [a, b], alpha and optional bars")
        names_pair = entry["players"]
        if not isinstance(names_pair, list) or len(names_pair) != 2 or names_pair[0] == names_pair[1]:
            raise ValueError("room.pair players must be two different players")
        a, b = (player(p, "room.pair player") for p in names_pair)
        alpha = entry["alpha"]
        a_from_b, b_from_a = (alpha if isinstance(alpha, list) else [alpha, alpha])
        first, last = _bars(entry["bars"], "room.pair") if "bars" in entry else (1, 1 << 30)
        for value in (a_from_b, b_from_a):
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError("room.pair alpha must be between 0 and 1")
        pairs.append((a, b, float(a_from_b), first, last))
        pairs.append((b, a, float(b_from_a), first, last))

    feels: dict[int, tuple[str, dict[str, Any]]] = {}
    raw_feels = table.get("feel", {})
    if not isinstance(raw_feels, dict):
        raise ValueError("room.feel must be a table of player or track = profile")
    # players first, then tracks, so a track's own feel wins
    ordered = sorted(raw_feels.items(), key=lambda item: item[0] not in players)
    for selector, spec in ordered:
        profile = _profile(spec, selector)
        tracks = players[selector] if selector in players else resolve(selector)
        for index in tracks:
            if index not in player_of:
                raise ValueError(f"room.feel {selector!r} names a track outside the room")
            feels[index] = (f"{player_of[index]}:{selector}", profile)
    raw_machines = table.get("machine", [])
    machines = frozenset(player(name, "room.machine") for name in
                         (raw_machines if isinstance(raw_machines, list) else [raw_machines]))
    if any(index in feels for name in machines for index in players[name]):
        raise ValueError("a room machine has no feel")
    return Room(players, player_of, lead, leads, pairs, params, feels, list(players), passages, feel_spans,
                machines)


def _profile(spec, selector) -> dict[str, Any]:
    if isinstance(spec, str):
        if spec not in FEELS:
            raise ValueError(f"unknown feel {spec!r} for {selector!r}; choices: {', '.join(FEELS)}")
        return FEELS[spec]
    if not isinstance(spec, dict):
        raise ValueError(f"room.feel {selector!r} must be a profile name or table")
    unknown = set(spec) - FEEL_KEYS
    if unknown:
        raise ValueError(f"unknown feel keys for {selector!r}: {sorted(unknown)}")
    profile = {**FEELS["straight"], **spec}
    if int(profile["subdivision"]) <= 0:
        raise ValueError("feel subdivision must be positive")
    for key in ("timing_ms", "velocity"):
        if not isinstance(profile[key], list) or not profile[key]:
            raise ValueError(f"feel {key} must be a nonempty array")
    return profile


def beat_grid(bars, score_end: int) -> list[int]:
    """Beat ticks from the start through the first beat at or after the score's end."""
    beats, bar = [], 1
    while True:
        start, end = bars.bar_start(bar), bars.bar_start(bar + 1)
        length = bars.beat_length_at(start)
        tick = start
        while tick < end:
            beats.append(tick)
            if tick >= score_end:
                return beats
            tick += length
        bar += 1


@dataclass
class Night:
    """What the room did: per-player beat displacements (s) and period deviations."""
    beats: list[int]
    target: list[float]                 # T(n), seconds
    onset: dict[str, list[float]]       # S(n) + x + M offsets, seconds
    phase: dict[str, list[float]]       # x_i(n), seconds
    period: dict[str, list[float]]      # p_i(n)
    plays: dict[str, list[bool]]


def simulate(room: Room, plays: dict[str, list[bool]], target: list[float], bar_of: list[int],
             rng: random.Random, bias=None, desired=None, audible=None) -> tuple[dict, dict, dict]:
    """Run the coupled timekeepers over the beat intervals `target`.

    `bias` and `desired` are performed-attack and intended-feel offsets in seconds.
    `audible` includes pulse cues from the completed beat's offbeat strokes.
    `plays[i][n]` says whether player i strikes beat n. Returns (onset, phase,
    period) per player, one entry per beat including the one closing the last interval.
    """
    p = room.params
    order = room.order
    beats = len(target)
    x = {i: 0.0 for i in order}
    rate = {i: 0.0 for i in order}
    recent = {i: 0.0 for i in order}
    onset = {i: [] for i in order}
    phase = {i: [] for i in order}
    period = {i: [] for i in order}
    mean_error = {i: 0.0 for i in order}
    memory = 1.0 / p["tempo_memory"]
    bound = p["drift_bound"]
    for n in range(beats):
        innovation, motor_factor, listening = room.timing_factors(bar_of[n])
        motor = {i: rng.gauss(0.0, p["motor_ms"] / 1000.0 * motor_factor) for i in order}
        for i in room.machines:
            motor[i] = 0.0
        raw = {i: x[i] + (motor[i] if plays[i][n] else 0.0) for i in order}
        mean_weight = room.mean_weight(bar_of[n])
        heard = {i: raw[i] + (mean_weight*bias[i][n] if bias is not None else 0.0) for i in order}
        intent = {i: mean_weight*desired[i][n] if desired is not None else 0.0 for i in order}
        sources = {i: 1.0 if plays[i][n] else mean_weight if audible is not None and audible[i][n] else 0.0
                   for i in order}
        for i in order:
            onset[i].append(raw[i])
            phase[i].append(x[i])
            period[i].append(rate[i])
            if n and target[n - 1] > 0:
                produced = (onset[i][n] - onset[i][n - 1]) / target[n - 1]
                recent[i] += memory * (produced - recent[i])
        bar = bar_of[n]
        step = {}
        for i in order:
            total = sum(room.alpha(i, j, bar)*sources[j] for j in order if j != i and sources[j])
            # Budget extra listening without reducing the authored baseline gain.
            gain = listening
            if listening > 1 and total:
                gain = max(1.0, min(listening, 0.85 / total))
            correction = sum(room.alpha(i, j, bar) * sources[j] * (heard[i] - heard[j] - intent[i] + intent[j])
                             for j in order if j != i and sources[j]) * gain
            others = [recent[j] for j in order if j != i]
            ensemble = sum(others) / len(others) if others else 0.0
            interval = target[n] * (1.0 + rate[i]) + rng.gauss(0.0, p["timekeeper"] * target[n] * innovation)
            step[i] = interval - target[n] - correction
            walk = rate[i] + rng.gauss(0.0, p["drift"] * innovation) - p["drift_return"] * rate[i] \
                - p["beta"] * listening * (rate[i] - ensemble)
            lead = room.lead_at(bar)
            if mean_weight and i != lead and sources[lead] and room.alpha(i, lead, bar) > 0 and target[n] > 0:
                error = heard[i] - heard[lead] - intent[i] + intent[lead]
                mean_error[i] += memory * (error - mean_error[i])
                # Integral period correction eliminates a steady phase error,
                # rather than requiring one to compensate a persistent rate error.
                strength = mean_weight * sources[lead] * min(1.0, room.alpha(i, lead, bar) / DEFAULTS['follow'])
                walk -= strength * max(0.0, listening - 1.0) / 6.0 * mean_error[i] / target[n]
            else:
                mean_error[i] *= 1.0 - memory
            rate[i] = max(-bound * innovation, min(bound * innovation, walk))
            if i in room.machines:
                step[i], rate[i] = 0.0, 0.0
        for i in order:
            x[i] += step[i]
    for i in order:
        onset[i].append(x[i])
        phase[i].append(x[i])
        period[i].append(rate[i])
    return onset, phase, period


class Feel:
    """One night's feel profile for one player's hands, with its slow wander."""

    def __init__(self, profile: dict[str, Any], rng: random.Random):
        self.subdivision = int(profile["subdivision"])
        self.rng = rng
        scale = 1.0 + float(profile["night_scale"]) * (2.0 * rng.random() - 1.0)

        def night(values, sd):
            return [scale * float(v) + max(-2 * sd, min(2 * sd, rng.gauss(0.0, sd))) if sd else scale * float(v)
                    for v in values]

        self.timing = night(profile["timing_ms"], float(profile["night_ms"]))
        self.velocity = night(profile["velocity"], float(profile["night_velocity"]))
        self.wander_sd = (float(profile["wander_ms"]), float(profile["wander_velocity"]))
        self.wander: list[tuple[list[float], list[float]]] = []

    def _wander(self, bar: int):
        keep = WANDER_CORRELATION
        kick = math.sqrt(1.0 - keep * keep)
        while len(self.wander) < bar:
            if self.wander:
                last_t, last_v = self.wander[-1]
            else:
                last_t = [self.rng.gauss(0.0, self.wander_sd[0]) for _ in self.timing]
                last_v = [self.rng.gauss(0.0, self.wander_sd[1]) for _ in self.velocity]
            self.wander.append((
                [keep * w + kick * self.rng.gauss(0.0, self.wander_sd[0]) for w in last_t],
                [keep * w + kick * self.rng.gauss(0.0, self.wander_sd[1]) for w in last_v],
            ))
        t, v = self.wander[bar - 1]
        cap_t, cap_v = 2 * self.wander_sd[0], 2 * self.wander_sd[1]
        return ([max(-cap_t, min(cap_t, w)) for w in t], [max(-cap_v, min(cap_v, w)) for w in v])

    def at(self, bars, tick: int) -> tuple[float, float]:
        """(timing ms, velocity) at a grid position; zero between grid points."""
        bar = bars.bar_at(tick)
        grid = bars.ppq * 4.0 / self.subdivision
        relative = tick - bars.bar_start(bar)
        position = round(relative / grid)
        if abs(relative - position * grid) > 1.0:
            return 0.0, 0.0
        wander_t, wander_v = self._wander(bar)
        k_t, k_v = position % len(self.timing), position % len(self.velocity)
        return self.timing[k_t] + wander_t[k_t], self.velocity[k_v] + wander_v[k_v]


def timing(room: Room, onsets, bars, tempo, warper, score_end: int, seed: int, compose=None):
    """Onset offsets (ms) and feel velocities for every room note; and the Night."""
    beats = beat_grid(bars, score_end)
    seconds_per_tick = [tempo.at(b) / (bars.ppq * 1_000_000) for b in beats]
    target = [(warper.apply(b1) - warper.apply(b0)) * spt
              for b0, b1, spt in zip(beats, beats[1:], seconds_per_tick)]
    beat_index = {tick: n for n, tick in enumerate(beats)}
    plays = {i: [False] * len(beats) for i in room.order}
    for note in onsets:
        who = room.player_of.get(note.track)
        if who is not None and note.tick in beat_index:
            plays[who][beat_index[note.tick]] = True
    bar_of = [bars.bar_at(b) for b in beats[:-1]]
    motor_rng = _stream(seed, "room-motor")
    feel_rng = _stream(seed, "room-feel")
    feels: dict[str, Feel] = {}
    for track in sorted(room.feels):
        key, profile = room.feels[track]
        if key not in feels:
            feels[key] = Feel(profile, random.Random(feel_rng.getrandbits(64)))
    feel_ms = {id(note): 0.0 for note in onsets}
    feel_velocity = {}
    by_tick = defaultdict(list)
    for note in sorted(onsets, key=lambda e: (e.tick, e.track, e.order)):
        if note.track not in room.player_of:
            continue
        if note.track in room.feels:
            ms, velocity = feels[room.feels[note.track][0]].at(bars, note.tick)
            feel_ms[id(note)] = ms
            feel_velocity[id(note)] = velocity
        by_tick[note.tick].append(note)
    for tick, notes in by_tick.items():
        weight = room.mean_weight(bars.bar_at(tick))
        if not weight:
            continue
        per_player = defaultdict(list)
        for note in notes:
            per_player[room.player_of[note.track]].append(feel_ms[id(note)])
        means = [sum(values)/len(values) for values in per_player.values()]
        spread = max(means) - min(means)
        limit = room.feel_span_at(bars.bar_at(tick))
        if spread > limit:
            center = sum(means)/len(means)
            factor = 1.0-weight + weight*limit/spread
            for note in notes:
                feel_ms[id(note)] = center + factor*(feel_ms[id(note)]-center)
    styled = compose(feel_ms.copy()) if compose is not None else feel_ms.copy()
    bias = {p: [0.0] * len(beats) for p in room.order}
    desired = {p: [0.0] * len(beats) for p in room.order}
    last_bias, last_desired = dict.fromkeys(room.order, 0.0), dict.fromkeys(room.order, 0.0)
    grouped = defaultdict(list)
    for note in onsets:
        if note.track in room.player_of:
            n = max(0, bisect.bisect_right(beats, note.tick)-1)
            grouped[(room.player_of[note.track], n)].append(note)
    audible = {p: [False] * len(beats) for p in room.order}
    for n in range(len(beats)):
        for player in room.order:
            # Offbeat strokes reveal a pulse too. Use only the completed beat,
            # plus attacks exactly on this beat; never listen to future notes.
            notes = grouped.get((player, n-1), []) + [note for note in grouped.get((player,n), []) if note.tick == beats[n]]
            if notes:
                last_bias[player] = sum(styled[id(note)] for note in notes)/len(notes)/1000.0
                last_desired[player] = sum(feel_ms[id(note)] for note in notes)/len(notes)/1000.0
                audible[player][n] = True
            bias[player][n], desired[player][n] = last_bias[player], last_desired[player]
    onset, phase, period = simulate(room, plays, target, bar_of, _stream(seed, 'room-time'), bias, desired, audible)
    motor_sd = room.params["motor_ms"]
    group_motor: dict[tuple[str, int], float] = {}
    offsets: dict[int, float] = {}
    for note in sorted(onsets, key=lambda e: (e.tick, e.track, e.order)):
        who = room.player_of.get(note.track)
        if who is None:
            continue
        n = min(len(beats) - 2, max(0, bisect.bisect_right(beats, note.tick) - 1))
        if note.tick in beat_index:
            seconds = onset[who][beat_index[note.tick]]
        else:
            fraction = (note.tick - beats[n]) / (beats[n + 1] - beats[n])
            if (who, note.tick) not in group_motor:
                group_motor[(who, note.tick)] = motor_rng.gauss(
                    0.0, motor_sd * room.timing_factors(bars.bar_at(note.tick))[1])
                if who in room.machines:
                    group_motor[(who, note.tick)] = 0.0
            seconds = (1 - fraction) * phase[who][n] + fraction * phase[who][n + 1] \
                + group_motor[(who, note.tick)] / 1000.0
        ms = 1000.0 * seconds
        offsets[id(note)] = ms + feel_ms[id(note)]
        feel_velocity.setdefault(id(note), 0.0)
    return offsets, feel_velocity, Night(beats, target, onset, phase, period, plays)


def dynamics(room: Room, onsets, pairs, event_ticks, base_velocity: dict[int, float],
             bars, night: Night) -> dict[int, float]:
    """Velocity offsets from loudness contagion, yield under the lead and answer in its gaps."""
    p = room.params
    beats = night.beats
    order = room.order
    per_beat: dict[str, dict[int, list[float]]] = {i: defaultdict(list) for i in order}
    index_of: dict[int, int] = {}
    for note in onsets:
        who = room.player_of.get(note.track)
        if who is None:
            continue
        n = max(0, bisect.bisect_right(beats, note.tick) - 1)
        index_of[id(note)] = n
        per_beat[who][n].append(base_velocity[id(note)])

    # A player's loudness is its swell: a two-beat average of its intended velocity
    # against its own sixteen-beat level, so phrase dynamics spread and single
    # accents don't. The others hear it a beat later.
    short = {i: None for i in order}
    long = {i: None for i in order}
    loud = {i: 0.0 for i in order}
    contagion = {i: 0.0 for i in order}
    lift: dict[str, list[float]] = {i: [] for i in order}
    cap = p["contagion_cap"]
    for n in range(len(beats)):
        bar = bars.bar_at(beats[n])
        new = {}
        for i in order:
            weights = {j: room.alpha(i, j, bar) for j in order if j != i}
            total = sum(weights.values())
            pull = sum(w * (loud[j] - loud[i]) for j, w in weights.items()) / total if total else 0.0
            new[i] = max(-cap, min(cap, p["contagion_memory"] * contagion[i] + p["contagion"] * pull))
        for i in order:
            contagion[i] = new[i]
            lift[i].append(contagion[i])
            values = per_beat[i].get(n)
            if values:
                mean = sum(values) / len(values)
                short[i] = mean if short[i] is None else short[i] + (mean - short[i]) / 2.0
                long[i] = mean if long[i] is None else long[i] + (mean - long[i]) / 16.0
        loud = {i: (short[i] - long[i] if short[i] is not None else 0.0) + contagion[i] for i in order}

    # the lead's line: its voice tracks' notes, by passage
    line: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for note in onsets:
        off = event_ticks.get(pairs.get(id(note)), note.tick + 1)
        line[note.track].append((note.tick, off, note.message.note))
    for notes in line.values():
        notes.sort()

    result: dict[int, float] = {}
    for note in onsets:
        who = room.player_of.get(note.track)
        if who is None:
            continue
        v = lift[who][index_of[id(note)]]
        bar = bars.bar_at(note.tick)
        lead = room.lead_at(bar)
        if who != lead:
            beat = bars.beat_length_at(note.tick)
            voice = sorted(n for t in room.voice_at(bar) for n in line[t])
            active = [pitch for on, off, pitch in voice if on <= note.tick < off and note.tick - on < beat]
            if active and min(abs(pitch - note.message.note) for pitch in active) <= p["yield_semitones"]:
                v -= p["yield"]
            else:
                # the line stops moving a beat after its last note; a gap lasts until its next
                starts = [on for on, _, _ in voice]
                k = bisect.bisect_right(starts, note.tick)
                if k and starts[k - 1] < note.tick:
                    opened = starts[k - 1] + beat
                    closes = starts[k] if k < len(starts) else math.inf
                    if note.tick >= opened and closes - opened >= p["gap_beats"] * beat:
                        taper = 1.0 - (note.tick - opened) / (2.0 * beat)
                        v += p["answer"] * max(0.0, taper)
        result[id(note)] = v
    return result
