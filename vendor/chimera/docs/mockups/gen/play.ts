/** PLAY mode mockup: an adult male holding /ɑ/ at 110 Hz, tongue being dragged from /i/. → out/play.html */
import { HumanVoice } from "../../../src/engine/voice/humanVoice.ts";
import { BODY_MALE } from "../../../src/engine/anatomy/human.ts";
import { VoiceModel, buildFaceData, spectrumData, tubeData, vowelArtic, lerpArtic } from "./face.ts";
import { C, topBar, faceScreen, tubeScreen, foldsScreen, spectrumScreen, vowelsScreen, encoderScreen, keysScreen, writePage } from "./lib.ts";

const FS = 48000, F0 = 110;
const artic = vowelArtic("male", "ɑ");
const from = vowelArtic("male", "i");
const model = new VoiceModel("male", FS);
const voice = new HumanVoice({ fsOut: FS, body: BODY_MALE });
voice.setSource({ f0: F0, pressure: 8, breathiness: 0.15, tension: 0.5, vibratoDepth: 0 });
voice.setArticulation(artic);
voice.setGate(1);
const total = Math.round(FS * 0.9);
const out = new Float32Array(total);
for (let i = 0; i < total; i += 128) {
  voice.render(out, i, Math.min(128, total - i));
  if (i === Math.round((FS * 0.75) / 128) * 128) voice.telemetry(); // reset the accumulators
}
const tele = voice.telemetry();
const F = buildFaceData(model, artic, tele, { ghosts: [0.62, 0.76, 0.88].map((u) => lerpArtic(from, artic, u)) });
const spec = spectrumData(out, FS, tele.f0, model, artic);
const trail = [0, 0.2, 0.4, 0.6, 0.8, 1].map((u) => { const t = model.transfer(lerpArtic(from, artic, u)); return [t.formants[0], t.formants[1]] as [number, number]; });

const parts = [
  topBar("PLAY"),
  faceScreen({ x: 16, y: 52, w: 700, h: 832 }, F, {
    slots: [
      { label: "TONGUE", value: "ɑ", color: C.green, ipa: true },
      { label: "JAW", value: artic.jaw.toFixed(1), unit: "CM", color: C.red },
      { label: "LIPS", value: artic.lipAp.toFixed(1), unit: "CM", color: C.pink },
      { label: "VELUM", value: "SHUT", color: C.white },
    ],
  }),
  tubeScreen({ x: 728, y: 52, w: 342, h: 200 }, tubeData(model, tele)),
  foldsScreen({ x: 1082, y: 52, w: 342, h: 200 }, Array.from(tele.glottalArea), Array.from(tele.glottalFlow)),
  spectrumScreen({ x: 728, y: 264, w: 696, h: 200 }, spec.harmonics, spec.envelope, spec.formants, tele.f0),
  vowelsScreen({ x: 728, y: 476, w: 284, h: 224 }, model.vowels, trail, [F.formants[0], F.formants[1]], "ɑ"),
  encoderScreen({ x: 1024, y: 476, w: 400, h: 224 }, "VOICE", [
    { label: "PITCH", value: String(Math.round(tele.f0)), unit: "HZ  A2", frac: Math.log(tele.f0 / 40) / Math.log(1100 / 40), color: C.green },
    { label: "BREATH", value: tele.pLung.toFixed(1), unit: "CMH2O", frac: (tele.pLung - 2) / 20, color: C.red },
    { label: "PRESSED", value: "50", unit: "%", frac: 0.5, color: C.pink },
    { label: "VIBRATO", value: "0", unit: "CENTS", frac: 0, color: C.white },
  ]),
  keysScreen({ x: 728, y: 712, w: 696, h: 172 }, "KEYS", "A2  110 HZ", 45),
];
const path = await writePage("play", "CHIMERA Play mockup", parts, "Play: an adult male holding /ɑ/ at 110 Hz, 8 cmH2O, the tongue dragged from /i/.");
console.log("wrote", path);
