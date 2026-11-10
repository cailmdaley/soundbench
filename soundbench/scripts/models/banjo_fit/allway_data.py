#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Pin distinct Philharmonia performances into train, development and test splits."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
# Every string has training coverage; string identities are assumed model labels.
BANKS = {
    'train': [('D3',50,4,'forte'),('G3',55,3,'forte'),('A3',57,3,'forte'),('B3',59,2,'forte'),('D4',62,1,'forte'),('E4',64,1,'forte'),('G4',67,5,'piano'),('A4',69,1,'forte'),('B4',71,1,'forte')],
    'development': [('E3',52,4,'forte'),('As3',58,3,'forte'),('C4',60,2,'forte'),('Ds4',63,1,'forte')],
    'test_notes': [('Fs3',54,4,'forte'),('Gs3',56,3,'forte'),('Cs4',61,2,'forte'),('F4',65,1,'forte')],
    'test_dynamics': [('D3',50,4,'piano'),('G3',55,3,'piano'),('A3',57,3,'piano'),('B3',59,2,'piano'),('D4',62,1,'piano'),('E4',64,1,'piano'),('A4',69,1,'piano'),('B4',71,1,'piano')],
}


def prepare(directory):
    from soundbench.scripts.models.banjo_fit.fetch import SHA256, URL
    archive = ROOT/'references/audio/fit/philharmonia/Strings.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('archive checksum mismatch')
    directory.mkdir(parents=True, exist_ok=True)
    identities = set()
    result = {}
    with zipfile.ZipFile(archive) as package:
        for split, rows in BANKS.items():
            notes = []
            for name, pitch, string, dynamic in rows:
                stem = f'banjo_{name}_very-long_{dynamic}_normal'
                member = f'Strings/banjo/{stem}.mp3'
                payload = package.read(member)
                sha = hashlib.sha256(payload).hexdigest()
                if sha in identities:
                    raise ValueError('a source performance crosses split boundaries')
                identities.add(sha)
                mp3 = directory/(stem+'.mp3')
                mp3.write_bytes(payload)
                wav = mp3.with_suffix('.wav')
                subprocess.run(['ffmpeg','-v','error','-y','-i',str(mp3),'-ar','48000',str(wav)],check=True)
                notes.append(dict(path=wav.name,pitch=pitch,string=string,dynamic=dynamic,
                                  source_url=URL,source_member=member,sha256=sha,split=split,
                                  string_assignment='assumed; not documented by source'))
            destination = directory/(split+'.json')
            destination.write_text(json.dumps(notes,indent=2)+'\n')
            result[split] = str(destination)
    (directory/'split.json').write_text(json.dumps(dict(archive_sha256=SHA256,manifests=result,
        policy='Development pitches choose methods; test pitches and alternate-dynamic source files never enter optimization. Distinct hashes do not establish recording-session independence.'),indent=2)+'\n')
    return result

if __name__ == '__main__':
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    print(json.dumps(prepare(ROOT/'build/banjo-allway/data'),indent=2))
