# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "scipy>=1.11", "numba>=0.61"]
# ///
"""The actual coupled kernel must sound two distinct, sequentially picked strings."""
import copy
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from soundbench.scripts.models import guitar
from soundbench.scripts.models.folk_strings import load_config, note, tremolo
from soundbench.scripts.models.course_diagnostics import partial_envelopes, fundamental, SR


class Courses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=load_config('mandolin')
        cls.c['engine']['output_gain']=1.
        cls.c['excitation']['slide_noise']=0.
        cls.rows=note(0,2.5,76,90,1,10)
        cls.members=guitar.render(cls.rows,3,config=cls.c,member_stems=True)

    def test_actual_lane_onsets_positions_and_amplitudes(self):
        for direction in ('down','up'):
            with patch.object(guitar,'_render',wraps=guitar._render) as kernel:
                audio=guitar.render(note(0,.5,76,90,1,10,direction=direction),.8,config=self.c,member_stems=True)
            rows,pulses=kernel.call_args.args[:2]
            picked=np.flatnonzero(rows[:,1]==0)
            self.assertEqual(set(rows[picked,2]),{0,4})
            primary=next(i for i in picked if rows[i,2]==0)
            secondary=next(i for i in picked if rows[i,2]==4)
            expected=round(self.c['excitation']['course_delay_ms']*SR/1000)
            self.assertEqual(rows[secondary,0]-rows[primary,0],expected if direction=='down' else -expected)
            self.assertGreater(abs(rows[primary,8]-rows[secondary,8]),.006)
            combs=[int(rows[i,8]*SR/(440*2**((rows[i,4]-69)/12))) for i in (primary,secondary)]
            self.assertNotEqual(combs[0],combs[1])
            self.assertAlmostEqual(np.sum(pulses[secondary])/np.sum(pulses[primary]),.46/.54)
            for member,index in enumerate((primary,secondary)):
                nonzero=np.flatnonzero(abs(audio[member])>1e-15)
                self.assertEqual(nonzero[0],rows[index,0])
            self.assertTrue(np.isfinite(audio).all())
            self.assertLess(abs(audio).max(),.98)

    def test_diagnostic_members_sum_to_production_render(self):
        normal=guitar.render(self.rows,3,config=self.c)
        np.testing.assert_allclose(self.members.sum(axis=0),normal[0],rtol=1e-10,atol=1e-12)

    def test_independent_tuning_and_no_doubled_copy(self):
        for course,pitch in enumerate(self.c['strings']['open_midi'],1):
            audio=guitar.render(note(0,2,pitch,90,course,10),2.5,config=self.c,member_stems=True)
            hz=[fundamental(y,pitch) for y in audio]
            cents=1200*np.log2(hz[1]/hz[0])
            self.assertGreater(cents,1.)
            self.assertLess(cents,4.)
            scale=np.dot(audio[0],audio[1])/np.dot(audio[0],audio[0])
            residual=np.linalg.norm(audio[1]-scale*audio[0])/np.linalg.norm(audio[1])
            self.assertGreater(residual,.2)
            print('course',course,'measured pair detuning',cents,'cents; non-copy residual',residual)

    def test_sustained_partial_beating(self):
        for h in (1,2,3):
            env,ratio=partial_envelopes(self.members,76,h)
            t=np.arange(len(ratio))/SR
            use=(t>.08)&(t<1.25)&(env.sum(axis=0)>env.max()*.01)
            self.assertLess(ratio[use].min(),.35)
            self.assertGreater(ratio[use].max(),.85)
            print('partial',h,'interference ratio',ratio[use].min(),ratio[use].max())

    def test_weak_coupling_actually_transfers_energy(self):
        coupled=guitar.render(self.rows,3,config=self.c,member_stems=True,excite_member=0)
        c=copy.deepcopy(self.c);c['bridge']['coupling']=0.
        isolated=guitar.render(self.rows,3,config=c,member_stems=True,excite_member=0)
        self.assertEqual(abs(isolated[1]).max(),0.)
        self.assertGreater(np.linalg.norm(coupled[1]),1e-8)
        ratio=np.linalg.norm(coupled[1])/np.linalg.norm(coupled[0])
        self.assertGreater(ratio,1e-5)
        self.assertLess(ratio,.05)
        print('passive receiver/driver RMS ratio',ratio)

    def test_alternating_tremolo_and_release(self):
        rows=tremolo(0,1,76,80,1,20,evenness=1)
        directions=[int(r[7]) for r in rows if r[1]==0]
        self.assertEqual(directions,[i%2 for i in range(len(directions))])
        a=guitar.render(rows,2.5,config=self.c,member_stems=True)
        b=guitar.render(rows,2.5,config=self.c,member_stems=True)
        self.assertTrue(np.array_equal(a,b))
        self.assertLess(np.sqrt(np.mean(a[:,-4800:]**2)),1e-6)


if __name__=='__main__':unittest.main()
