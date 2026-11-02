import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import collider_event_deflections as d

def fixture():
    # Source identities deliberately differ from later tracker iteration order.
    text='5 125 0 0 0 .01745 0 .99984\n5 -125 0 0 0 -.01745 0 -.99984\n10 125 0 0 0 0 .01745 .99984\n10 -125 0 0 0 0 -.01745 -.99984\n'
    data=text.encode();ref=dict(path='event.ini',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    identity=dict(schema_version=d.IDENTITY_SCHEMA,coordinate_frame='native_pair_tracking',source_artifact=copy.deepcopy(ref),events=[dict(event_id=i,particles=[dict(source_record=2*i+1,event_slot='electron',role='electron',pdg_id=11),dict(source_record=2*i+2,event_slot='positron',role='positron',pdg_id=-11)]) for i in range(2)])
    records=d.parse_input(data);initial=[[i+1,i+1,row[0],row[0],*row[1:]] for i,row in enumerate(records)]
    flags=dict(load_events=1,track_pairs=1,do_pairs=0,beam_pair=0,do_comp=0,do_prod=0,do_bhabhas=0,pair_ratio=1,pair_ecut=.001,charge_sign=-1)
    initial_current=[[r[0],*r[8:11],0,0,0] for r in initial]
    current=copy.deepcopy(initial_current);current[0][1]+=.0001;current[1][2]+=.0002
    current[0][4:]=[3,2,1];current[1][4:]=[2,1,0]
    current.reverse()
    raw=dict(schema_version=d.RAW_SCHEMA,available=True,reason=None,phase='post-diagonal-tracker-state',coordinate_frame='native_pair_tracking',source_file='event.ini',
        initial_columns=d.INITIAL_COLUMNS,current_columns=d.CURRENT_COLUMNS,evaluated_flags=flags,initial_records=initial,
        frames=[dict(id=0,diagonal=0,records=[]),dict(id=1,diagonal=1,records=initial_current[:2]),dict(id=2,diagonal=2,records=current)],final=dict(records=current))
    manifest=dict(imported_trajectories_file='gui_imported_trajectories.json',native_beam_angles_x_rad=[0,0],native_beam_angles_y_rad=[0,0],frames=[dict(id=i,diagonal=i,elapsed_s=i*1e-12) for i in range(3)])
    summary='\n'.join(k+' = '+str(v) for k,v in dict(flags,store_pairs=1).items())+'\n'
    return data,identity,records,ref,raw,manifest,summary

def write_fixture(root):
    data,identity,records,ref,raw,manifest,summary=fixture()
    (root/'event.ini').write_bytes(data)
    (root/'event_identity.json').write_text(json.dumps(identity))
    (root/'gui_imported_trajectories.json').write_text(json.dumps(raw))
    (root/'out.dat').write_text(summary)
    (root/'gui_manifest.fixture.json').write_text(json.dumps(manifest))
    result=d.build(root,manifest);d.write_analysis(root,result);return result

class TestEventDeflections(unittest.TestCase):
    def setUp(self):self.fixture=fixture()
    def make(self):
        data,identity,records,ref,raw,manifest,summary=self.fixture
        result=d.make_data(raw,identity,records,ref,manifest,raw['evaluated_flags'])
        result['source_artifacts']={k:ref for k in ('identity','trajectories','input','summary')}
        d.validate(result);return result
    def test_actual_key_join_empty_then_complete(self):
        result=self.make();self.assertEqual(len(result['plots']),4)
        self.assertEqual(result['frames'][0]['nrecords'],0)
        self.assertIsNone(result['frames'][0]['histograms']['electron_delta_theta']['statistics'])
        self.assertEqual(result['frames'][1]['event_counts'],dict(expected=2,complete=1,incomplete=0,not_yet_admitted=1))
        self.assertEqual(result['total']['event_counts'],dict(expected=2,complete=2,incomplete=0,not_yet_admitted=0))
        self.assertEqual(result['total']['field_gathers'],dict(main_mesh=5,outer_mesh=3,monopole=1))
        self.assertGreater(result['total']['histograms']['electron_delta_theta']['statistics']['mean'],0)
    def test_reordering_does_not_change_identity_statistics(self):
        expected=self.make()
        raw=self.fixture[4];raw['initial_records'].reverse()
        for frame in raw['frames']:frame['records'].reverse()
        raw['final']['records']=raw['frames'][-1]['records']
        self.assertEqual(self.make(),expected)
    def test_wrap_branch_crossing_and_exact_boundary(self):
        def v(phi):return [math.cos(math.radians(phi)),math.sin(math.radians(phi)),0]
        self.assertAlmostEqual(d.delta(v(179),v(-179))[1],2)
        self.assertAlmostEqual(d.delta(v(-179),v(179))[1],-2)
        self.assertEqual(d.delta(v(0),v(180))[1],-180)
    def test_zero_control(self):
        raw=self.fixture[4]
        final=[[r[0],*r[8:11],0,0,0] for r in raw['initial_records']]
        raw['frames'][-1]['records']=final;raw['final']['records']=final
        data=self.make()
        for h in data['total']['histograms'].values():
            self.assertEqual(h['statistics']['mean'],0);self.assertEqual(h['statistics']['variance'],0)
            self.assertEqual(h['samples'],2);self.assertEqual(sum(x['weight'] for x in h['bins']),2)
    def test_partial_event_is_explicit(self):
        raw=self.fixture[4];raw['frames'][1]['records']=raw['frames'][1]['records'][:1]
        self.assertEqual(self.make()['frames'][1]['event_counts']['incomplete'],1)
    def test_undefined_azimuth_excluded_without_fake_phi(self):
        raw=self.fixture[4];row=next(r for r in raw['frames'][-1]['records'] if r[0]==1);row[1:4]=[0,0,1]
        result=self.make();h=result['total']['histograms']['electron_delta_phi']
        self.assertEqual(h['excluded_undefined_azimuth'],1);self.assertEqual(h['samples'],1)
        self.assertEqual(result['total']['histograms']['electron_delta_theta']['samples'],2)
    def test_wrong_pdg_charge_refused(self):
        self.fixture[1]['events'][0]['particles'][0]['pdg_id']=-11
        with self.assertRaises(ValueError):self.make()
    def test_wrong_initial_row_and_float_refused(self):
        for index in (1,2,4,8):
            self.fixture=fixture();self.fixture[4]['initial_records'][0][index]+=1
            with self.subTest(index=index),self.assertRaises(ValueError):self.make()
    def test_duplicate_current_label_refused(self):
        self.fixture[4]['frames'][-1]['records'][0][0]=1
        with self.assertRaises(ValueError):self.make()
    def test_unknown_current_label_refused(self):
        self.fixture[4]['frames'][-1]['records'][0][0]=555
        with self.assertRaises(ValueError):self.make()
    def test_disappearing_identity_refused(self):
        self.fixture[4]['frames'][-1]['records'].pop()
        with self.assertRaises(ValueError):self.make()
    def test_gather_counter_regression_refused(self):
        self.fixture[4]['frames'][1]['records'][0][4]=999
        with self.assertRaises(ValueError):self.make()
    def test_final_differs_from_last_frame_refused(self):
        self.fixture[4]['final']['records']=[]
        with self.assertRaises(ValueError):self.make()
    def test_manifest_diagonal_mismatch_refused(self):
        self.fixture[5]['frames'][1]['diagonal']=9
        with self.assertRaises(ValueError):self.make()
    def test_crossing_angle_refused(self):
        self.fixture[5]['native_beam_angles_x_rad']=[.007,.007]
        with self.assertRaises(ValueError):self.make()
    def test_mixed_native_sources_refused(self):
        self.fixture[4]['evaluated_flags']['do_pairs']=1
        with self.assertRaises(ValueError):self.make()
    def test_hash_binding_refused(self):
        self.fixture[1]['source_artifact']['sha256']='a'*64
        with self.assertRaises(ValueError):self.make()
    def test_unsafe_slots_duplicate_rows_and_inconsistent_slots_refused(self):
        for mode in ('slot','row','missing'):
            self.fixture=fixture();p=self.fixture[1]['events'][1]['particles']
            if mode=='slot':p[0]['event_slot']='../electron'
            if mode=='row':p[0]['source_record']=1
            if mode=='missing':p.pop()
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.make()
    def test_input_malformed_refused(self):
        for source in (b'',b'\n',b'5 125 0 0 0 0 0\n',b'5 125 0 0 0 nan 0 1\n',b'5 125 0 0 0 0 0 0\n',b'5 125 0 0 0 0 0 1\n4 125 0 0 0 0 0 1\n'):
            with self.subTest(source=source),self.assertRaises(ValueError):d.parse_input(source)
    def test_nonnumeric_or_nonfinite_current_refused(self):
        for v in (math.nan,math.inf,'0',None):
            self.fixture=fixture();self.fixture[4]['frames'][-1]['records'][0][1]=v
            with self.subTest(v=v),self.assertRaises(ValueError):self.make()
    def test_derived_tampering_refused(self):
        for field in ('count','axis','sample','mode','event','gather','source'):
            result=self.make();h=result['total']['histograms']['electron_delta_theta']
            if field=='count':h['bins'][0]['weight']+=1
            if field=='axis':h['bins'][0]['low']-=.1
            if field=='sample':h['samples']+=1
            if field=='mode':h['statistics']['mode']=999
            if field=='event':result['total']['event_counts']['expected']+=1
            if field=='gather':result['total']['field_gathers']['monopole']=-1
            if field=='source':result['source_artifacts']['input']['path']='../input'
            with self.subTest(field=field),self.assertRaises(ValueError):d.validate(result)
    def test_build_write_and_source_authentication(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);result=write_fixture(p);self.assertTrue(d.validate(result))
            self.assertEqual(json.loads((p/'collider_event_deflections.json').read_text()),result)
            for ref in result['source_artifacts'].values():self.assertEqual(hashlib.sha256((p/ref['path']).read_bytes()).hexdigest(),ref['sha256'])
            (p/'event.ini').write_text((p/'event.ini').read_text().replace('125','124',1))
            with self.assertRaises(ValueError):d.build(p,self.fixture[5])
    def test_missing_identity_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            result=d.build(td);self.assertFalse(result['available']);self.assertTrue(d.validate(result))
    def test_missing_declared_native_source_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);write_fixture(p);(p/'gui_imported_trajectories.json').unlink()
            with self.assertRaises(ValueError):d.build(p,self.fixture[5])
    def test_duplicate_json_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'bad.json').write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):d.read_json(p,'bad.json')

if __name__=='__main__':unittest.main()
