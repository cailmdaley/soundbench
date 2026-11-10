#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Download checksum-pinned Philharmonia notes, decode and write a fit manifest."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
URL = 'https://philharmonia-assets.s3-eu-west-1.amazonaws.com/uploads/2020/02/12112005/Strings.zip'
SHA256 = '19157323ba3c5fe7970a9deadbff00165352996901be72d220319f0ff8e7aee5'
# String labels are model assignments, not recording metadata.
NOTES = [('D3', 50, 4), ('G3', 55, 3), ('A3', 57, 3), ('B3', 59, 2),
         ('D4', 62, 1), ('E4', 64, 1), ('G4', 67, 5), ('A4', 69, 1), ('B4', 71, 1)]


def main():
    directory = ROOT/'references/audio/fit/philharmonia'
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory/'Strings.zip'
    if not archive.exists():
        subprocess.run(['curl', '--fail', '-L', '-o', str(archive), URL], check=True)
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != SHA256:
            raise ValueError('Philharmonia archive checksum mismatch')
    manifest = []
    with zipfile.ZipFile(archive) as package:
        for name, pitch, string in NOTES:
            dynamic = 'piano' if name == 'G4' else 'forte'
            stem = f'banjo_{name}_very-long_{dynamic}_normal'
            mp3, wav = directory/f'{stem}.mp3', directory/f'{stem}.wav'
            mp3.write_bytes(package.read(f'Strings/banjo/{stem}.mp3'))
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(mp3), '-ar', '48000', str(wav)], check=True)
            manifest.append(dict(path=wav.name, pitch=pitch, string=string,
                                 source_url=URL, source_member=f'Strings/banjo/{stem}.mp3',
                                 sha256=hashlib.sha256(mp3.read_bytes()).hexdigest(),
                                 string_assignment='assumed; not documented by source'))
    (directory/'notes.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(directory/'notes.json')


if __name__ == '__main__':
    main()
