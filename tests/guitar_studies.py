"""Short diagnostic guitar studies; no sampled or delivery path."""
from soundbench.scripts.models.guitar import note, strum


def chord_releases(rows, boundaries):
    """Lift fretting fingers before changes; leave open strings undamped."""
    extra=[]
    opens=[64,59,55,50,45,40]
    for t in boundaries:
        current={}
        for row in sorted(rows,key=lambda r:r[0]):
            if row[0]>=round((t-.07)*48000):
                break
            if row[1] in (0,3,4):
                current[int(row[2])]=row
        for string,row in current.items():
            if row[4]!=opens[string]:
                extra.append([round((t-.07)*48000),1,string,row[3],0,0,0,0,2,0,row[10]])
    return rows+extra


def steel_demo(c):
    rows, token = [], 0
    beat = .75
    chords = {'G': ([67,59,55,50,47,43], [1,2,3,4,5,6]),
              'C': ([64,60,55,52,48], [1,2,3,4,5]),
              'D': ([66,62,57,50], [1,2,3,4])}
    for bar, chord in enumerate(['G','C','G','D','G','C','D','G']):
        t = bar*4*beat
        pitches, strings = chords[chord]
        bass = {'G':(43,6,50,4), 'C':(48,5,52,4), 'D':(50,4,45,5)}[chord]
        rows += note(t, t+.55, bass[0], [105,98,101,96,108,99,95,103][bar], bass[1], token)
        token += 1
        rows += note(t+2*beat, t+2*beat+.55, bass[2], 94, bass[3], token)
        token += 1
        for b in (1,3):
            voicing = [(p,s) for p,s in zip(pitches,strings) if s <= 3]
            rows += strum(t+b*beat, .5, [p for p,s in voicing], c,
                          velocity=80 if b == 1 else (53 if bar in (2,5) else 68),
                          strings=[s for p,s in voicing], muted=(b == 3 and bar in (1,4,6)),
                          tempo_bpm=80, accent=1.05 if b == 1 else .9, token=token)
            token += 24
        if bar in (3,7):
            # A short flatpicked bass turnaround instead of the last brush.
            for j,(p,s) in enumerate([(47,5),(45,5),(42,6),(43,6)]):
                on = t+(2.5+j*.375)*beat
                rows += note(on, on+.19, p, 91, s, token)
                token += 1
    return chord_releases(rows,[bar*3. for bar in range(1,8)]), 25.5


def nylon_demo(c):
    rows, token = [], 0
    chords = {'Am':([60,57,52,45],[2,3,4,5]), 'E':([59,56,52,47,40],[2,3,4,5,6]),
              'F':([60,57,53,48,41],[2,3,4,5,6])}
    melody = [[69,67,64],[64,65,67],[69,71,72],[71,68,64],
              [69,67,65],[65,64,65],[68,71,68],[69,64,69]]
    for bar, chord in enumerate(['Am','Am','F','E','Am','F','E','Am']):
        t = bar*3.
        p,s = chords[chord]
        rows += note(t, t+.75, p[-1], 94, s[-1], token, hardness=1)
        token += 1
        rows += strum(t+1, .45, p, c, velocity=round(78*[.95,.82,1.05,1.1,.9,.75,1.08,.88][bar]), strings=s,
                      speed_ms=24+bar%3*4, fingers=3 if bar not in (2,5) else 2, hardness=2, accent=1.08, token=token)
        token += 24
        rows += strum(t+2, .45, p, c, velocity=56, strings=s, direction='up', hardness=1 if bar in (1,5) else 2,
                      tempo_bpm=60, muted=bar == 5, token=token)
        token += 24
        for j,pitch in enumerate(melody[bar]):
            on = t+[0., 1.5, 2.5][j]
            rows += note(on, on+.4, pitch, 92 if j == 0 else 80, 1, token, hardness=2)
            token += 1
    return chord_releases(rows,[bar*3. for bar in range(1,8)]), 25.5
