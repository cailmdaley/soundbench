"""Score-addressed continuous touch and microphone-distance controls.

Velocity targets scale the complete performed velocity relative to the scored
velocity. Signed mic gain moves the direct field toward or away from the mic;
the room field is not boosted. Cut-only source gain shapes player projection
before both the direct and room paths. Controls interpolate across score bars.
"""
import bisect
import math
from soundbench.scripts.instrumentation import matching_tracks


def interpolate(points, position):
    if position <= points[0][0]:
        return points[0][1]
    i = bisect.bisect_right([p[0] for p in points], position)
    if i == len(points):
        return points[-1][1]
    a, b = points[i-1], points[i]
    return a[1] + (b[1]-a[1])*(position-a[0])/(b[0]-a[0])


def controls(settings, kind, indices, names, bars):
    stage = settings.get('stage', {})
    if set(stage)-{'velocity', 'mic', 'touch', 'gain'}:
        raise ValueError('unknown stage controls')
    result = {}
    for selector, rows in stage.get(kind, {}).items():
        matches = matching_tracks(selector, indices, names)
        if not matches:
            raise ValueError(f'stage selector {selector!r} has no track')
        points = []
        for row in rows:
            if len(row)==2: bar,value=row; beat=1
            elif len(row)==3: bar,beat,value=row
            else: raise ValueError('stage points require [bar,value] or [bar,beat,value]')
            if not isinstance(bar, int) or bar < 1 or not math.isfinite(value):
                raise ValueError('stage points require positive integer bars and finite values')
            if kind == 'velocity' and not 0 < value <= 127:
                raise ValueError('stage velocity must be in (0, 127]')
            if kind == 'gain' and not -24 <= value <= 0:
                raise ValueError('stage source gain must be in [-24, 0] dB')
            if kind == 'mic' and not -24 <= value <= 5:
                raise ValueError('stage mic gain must be in [-24, 5] dB')
            point = (bars.position(bar,beat), float(value))
            if points and point[0] <= points[-1][0]:
                raise ValueError('stage points must be strictly increasing')
            points.append(point)
        if not points:
            raise ValueError('stage controls require at least one point')
        for index in matches:
            if index in result:
                raise ValueError('duplicate stage track control')
            result[index] = points
    return result


def pitch_controls(settings, indices, names):
    """Cut-only picking-pressure registration, addressed by MIDI pitch."""
    result = {}
    for selector, rows in settings.get('stage', {}).get('touch', {}).items():
        matches = matching_tracks(selector, indices, names)
        if not matches:
            raise ValueError(f'stage touch selector {selector!r} has no track')
        points = []
        for pitch, scale in rows:
            if not isinstance(pitch,int) or not 0 <= pitch <= 127 or not math.isfinite(scale) or not .1 <= scale <= 1:
                raise ValueError('touch points require MIDI pitches and scale in [0.1, 1]')
            if points and pitch <= points[-1][0]:
                raise ValueError('touch pitches must be strictly increasing')
            points.append((pitch,float(scale)))
        if not points:
            raise ValueError('touch requires at least one point')
        for index in matches:
            if index in result:
                raise ValueError('duplicate touch selector')
            result[index] = points
    return result


def bar_beat(bars, tick):
    """Recover fractional notated beats without rounding a curve to its bar."""
    bar = bars.bar_at(tick)
    return bar, 1 + (tick-bars.bar_start(bar))/bars.beat_length_at(tick)


def volume_filter(points):
    """FFmpeg frame-evaluated gain; points are (seconds, direct gain in dB)."""
    expression = str(points[-1][1])
    for (a, x), (b, y) in reversed(list(zip(points, points[1:]))):
        line = f'{x}+({y-x})*(t-{a})/{b-a}'
        expression = f'if(lt(t,{b}),{line},{expression})'
    expression = f'if(lt(t,{points[0][0]}),{points[0][1]},{expression})'
    return f"volume='pow(10,({expression})/20)':eval=frame"
