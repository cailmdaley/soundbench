/**
 * SING mode mockup: the first demo song ("Made of air"), adult male, singer's ring on, caught
 * mid-note on "air" (/ɛ/ on C3) with the vibrato bloomed. → out/sing.html
 */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { HumanVoice, DEFAULT_SOURCE } from "../../../src/engine/voice/humanVoice.ts";
import { BODY_MALE } from "../../../src/engine/anatomy/human.ts";
import { EnglishG2P } from "../../../src/engine/phonetics/english.ts";
import { lyricsToSyllables } from "../../../src/engine/phonetics/syllables.ts";
import { planSong, midiToHz } from "../../../src/engine/phonetics/sing.ts";
import { DEMO_SONGS } from "../../../src/engine/phonetics/songs.ts";
import { REPO, VoiceModel, buildFaceData, spectrumData } from "./face.ts";
import { C, topBar, faceScreen, spectrumScreen, encoderScreen, keysScreen, writePage, screen, glow, line, text, rect, n1, dPath, textWidth, type Rect, type Pt } from "./lib.ts";
import { rollScreen, lyricsScreen, ringOverlay, nextKeyOverlay } from "./sing-screens.ts";

const FS = 48000;
const NOW = 5.55; // s on the song clock: 0.56 s into "air", vibrato fully bloomed
const VIBRATO = 35, VIB_RATE = DEFAULT_SOURCE.vibratoRate, RING = 0.85;

// ---------------------------------------------------------------- the song, planned as the app does
const g2p = new EnglishG2P();
g2p.setDictionaryText(gunzipSync(readFileSync(REPO + "/public/data/cmudict.txt.gz")).toString("utf8"));
const song = DEMO_SONGS[0];
const syl = lyricsToSyllables(song.lyrics, g2p);
const notes = song.notes.map((n, i) => ({ midi: n[0] + song.octave.male * 12, beats: n[1], syl: syl[i % syl.length] }));
const T0 = 0.35, beat = 60 / song.bpm;
const plan = planSong(notes, { body: "male", bpm: song.bpm, t0: T0, vibrato: VIBRATO });
let tt = T0;
const roll = notes.map((n) => { const r = { midi: n.midi, t0: tt, t1: tt + n.beats * beat, text: n.syl.text.toUpperCase() }; tt = r.t1; return r; });
const cur = roll.findIndex((r) => r.t0 <= NOW && NOW < r.t1);

// ---------------------------------------------------------------- render up to now, sampling telemetry
const voice = new HumanVoice({ fsOut: FS, body: BODY_MALE });
voice.setSource({ ...DEFAULT_SOURCE, f0: midiToHz(notes[0].midi) });
voice.setArticulation({ epilarynx: RING });
voice.schedule(plan.gestures);
const nNow = Math.round((NOW * FS) / 128) * 128;
const out = new Float32Array(nNow);
const trace: { t: number; m: number; v: number }[] = [];
const snaps: { t: number; artic: Float32Array }[] = [];
let tele = voice.telemetry();
for (let i = 0, blk = 0; i < nNow; i += 128, blk++) {
  voice.render(out, i, 128);
  if (blk % 6 === 5 && i + 128 < nNow) {
    tele = voice.telemetry();
    const t = (i + 128) / FS;
    trace.push({ t, m: tele.f0 > 20 ? 69 + 12 * Math.log2(tele.f0 / 440) : NaN, v: tele.voicing });
    snaps.push({ t, artic: tele.artic });
  }
}
tele = voice.telemetry(); // the last ~16 ms
trace.push({ t: NOW, m: 69 + 12 * Math.log2(tele.f0 / 440), v: tele.voicing });

const model = new VoiceModel("male", FS);
const artic = VoiceModel.articFromArray(tele.artic);
const ghostAt = (dt: number) => VoiceModel.articFromArray(snaps.reduce((b, s) => (Math.abs(s.t - (NOW - dt)) < Math.abs(b.t - (NOW - dt)) ? s : b)).artic);
const F = buildFaceData(model, artic, tele, { ghosts: [0.09, 0.06, 0.03].map(ghostAt) });
const spec = spectrumData(out, FS, midiToHz(roll[cur].midi), model, artic, 4096);
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const noteName = (m: number) => `${NAMES[m % 12]}${Math.floor(m / 12) - 1}`;
const curNote = roll[cur], next = roll[cur + 1];
console.log(`now ${NOW}s note ${cur} ${curNote.text} ${noteName(curNote.midi)} f0 ${tele.f0.toFixed(1)} voicing ${tele.voicing.toFixed(2)} formants ${F.formants.map(Math.round)} noise ${tele.noiseLevel.toFixed(1)} sparks ${F.sparks.length}`);

// ---------------------------------------------------------------- compose
const R = { x: 728, w: 696 };
const S = {
  face: { x: 16, y: 52, w: 700, h: 832 },
  roll: { x: R.x, y: 52, w: R.w, h: 320 },
  lyrics: { x: R.x, y: 384, w: R.w, h: 104 },
  spec: { x: R.x, y: 500, w: 342, h: 200 },
  voice: { x: 1082, y: 500, w: 342, h: 200 },
  keys: { x: R.x, y: 712, w: R.w, h: 172 },
};
const hz = Math.round(midiToHz(curNote.midi));
const parts = [
  topBar("SING"),
  faceScreen(S.face, F, {
    sub: "ADULT MALE / SINGING",
    active: null,
    info: [`NOTE ${noteName(curNote.midi)}  ${hz} HZ`, `SYLLABLE ${curNote.text}`, `RING ON`],
    slots: [
      { label: "TONGUE", value: "ɛ", color: C.green, ipa: true },
      { label: "JAW", value: artic.jaw.toFixed(1), unit: "CM", color: C.red },
      { label: "LIPS", value: artic.lipAp.toFixed(1), unit: "CM", color: C.pink },
      { label: "VELUM", value: "SHUT", color: C.white },
    ],
  }),
  rollScreen(S.roll, roll, trace, NOW, cur, T0, beat),
  lyricsScreen(S.lyrics, syl, cur, song.name.toUpperCase()),
  spectrumScreen(S.spec, spec.harmonics, spec.envelope, spec.formants, midiToHz(curNote.midi), { sub: "" }),
  ringOverlay(S.spec, spec.envelope),
  encoderScreen(S.voice, "VOICE", [
    { label: "VIBRATO", value: String(VIBRATO), unit: "CENTS", frac: VIBRATO / 80, color: C.green },
    { label: "RATE", value: VIB_RATE.toFixed(1), unit: "HZ", frac: (VIB_RATE - 3.5) / 4, color: C.red },
    { label: "BREATHY", value: String(Math.round(DEFAULT_SOURCE.breathiness * 100)), unit: "%", frac: DEFAULT_SOURCE.breathiness, color: C.pink },
    { label: "RING", value: String(Math.round(RING * 100)), unit: "%", frac: RING, color: C.white },
  ], 1, 1, "SUNG"),
  keysScreen(S.keys, "KEYS", `${noteName(curNote.midi)}  ${hz} HZ    NEXT ${next.text} ON ${noteName(next.midi)}`, curNote.midi, { above: new Map([[next.midi, next.text]]) }),
  nextKeyOverlay(S.keys, next.midi),
];
void [screen, glow, line, text, rect, n1, dPath, textWidth];
const path = await writePage("sing", "CHIMERA Sing mockup", parts, `Sing: "${song.name}", adult male, singer's ring on, ${NOW} s in: "air" (/ɛ/) on C3 with ${VIBRATO}-cent vibrato.`);
console.log("wrote", path);
export type { Rect, Pt };
