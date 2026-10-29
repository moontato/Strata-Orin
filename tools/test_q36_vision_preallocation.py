"""CPU/MOCK ONLY. No encoder/model/service invoked by these tests."""
import array
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch
from tools import q36_vision_fixture as fixture
from tools import q36_vision_preallocation_gate as worker
from tools.q36_vision_baseline_export import overlay,BASE
ROOT=Path(__file__).resolve().parents[1]

def block(text,marker):
    at=text.index(marker);start=text.index('{',at);level=1;end=start+1
    while level:level+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[at:end]

def write_sve(path,n,nx,ny,nonfinite=False):
    values=array.array('f',[0.5]*(n*2048))
    if nonfinite:values[0]=float('nan')
    Path(path).write_bytes(struct.pack('<5i',0x31455653,n,nx,ny,2048)+values.tobytes())

class Vision(unittest.TestCase):
    def test_encoder_header_finite_and_width_checks_are_not_mock_gpu_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.sve';write_sve(path,1024,32,32)
            self.assertEqual(fixture.sve(path)['delta'],-992)
            write_sve(path,1024,32,32,True)
            with self.assertRaisesRegex(ValueError,'nonfinite'):fixture.sve(path)
            path.write_bytes(struct.pack('<5i',0x31455653,1,1,1,4)+bytes(16))
            with self.assertRaisesRegex(ValueError,'2048'):fixture.sve(path)

    def test_public_images_and_fixture_geometry_with_mock_render_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture.make_images(root/'images')
            self.assertEqual((root/'images/red-square.png').read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            model=root/'model';model.touch();a=root/'a.sve';b=root/'b.sve';write_sve(a,1024,32,32);write_sve(b,512,32,16)
            class Tokenizer:
                @staticmethod
                def from_gguf(_):return Tokenizer()
                def encode(self,text,parse_special=False):
                    if text=='<|image_pad|>':return [248056]
                    result=[]
                    for i,part in enumerate(text.split('<|image_pad|>')):
                        if i:result.append(248056)
                        result.extend(part.encode())
                    return result
            class Template:
                def __init__(self,_):pass
                def render(self,messages,**kwargs):
                    return 'P'+''.join('<|image_pad|>' if x['type']=='image' else x['text'] for x in messages[0]['content'])
            with patch('tools.gguf_reader.GGUFFile',return_value=Mock(metadata={'tokenizer.chat_template':'MOCK ONLY'})),patch('tools.strata_tokenizer.Tokenizer',Tokenizer),patch('serve.frontend.ChatTemplate',Template):
                value=fixture.prepare(model,a,b,root/'prepared')
            self.assertEqual(value['image_a_pad_begin'],1);self.assertEqual(value['image_a_pad_end'],1025)
            self.assertGreater(value['tokens']['image_a'],1536);self.assertGreater(value['tokens']['text'],1024)
            self.assertEqual(value['image_b']['delta'],-480)
            self.assertTrue(a.exists() and b.exists())

    def test_protocol_checks_sampled_image_exclusion_and_text_offers(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'log'
            path.write_text('T 17\nDONE 2 100 1 2 length 0 0 0 0 0 0 0 0.0 100 0\nDONE 12 1200 2 3 length 3 8 0 0 0 0 0 0.0 1200 0\n')
            self.assertTrue(worker.protocol(path))
            path.write_text(path.read_text().replace('length 0 0 0','length 0 2 0'))
            with self.assertRaisesRegex(ValueError,'exclusion'):worker.protocol(path)
            path.write_text('ERR malformed input\n')
            with self.assertRaisesRegex(ValueError,'native error'):worker.protocol(path)

    def test_complete_stream_comparison_not_hash_or_marker_only(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b';a.write_bytes(b'a'*140000);b.write_bytes(a.read_bytes())
            self.assertTrue(worker.exact(a,b));b.write_bytes(b'a'*139999+b'b');self.assertFalse(worker.exact(a,b))

    def test_worker_reaps_only_its_owned_process_groups(self):
        process=Mock();process.pid=123456;process.poll.return_value=None
        process.wait.side_effect=[subprocess.TimeoutExpired('mock',20),0]
        with patch('os.killpg') as kill:worker.reap(process)
        self.assertEqual(kill.call_args_list[0].args,(123456,signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args,(123456,signal.SIGKILL));self.assertEqual(process.wait.call_count,2)

    def test_numerical_kernels_and_text_eligibility_source_unchanged(self):
        reference=subprocess.run(['git','show',BASE+':src/qwen35/q36_engine.cu'],cwd=ROOT,capture_output=True)
        if reference.returncode:self.skipTest('Optional source-history comparison requires the baseline commit')
        base=reference.stdout.decode();current=(ROOT/'src/qwen35/q36_engine.cu').read_text();diagnostic=overlay(base)
        for marker in ['template <class F> int64_t process(', 'void f32mm(', 'void layer(', 'void forward(', 'void forward_mtp(', 'void run_rows(', 'void mtp_rows(', 'void preallocate_kv(', 'void checkpoint_transfer(']:
            self.assertEqual(block(base,marker),block(current,marker),marker);self.assertEqual(block(base,marker),block(diagnostic,marker),marker)
        # Capture publication now retires metadata before writing, but payload
        # layout/copy, capacity, numerical methods and image eligibility stay fixed.
        capture=block(current,'void checkpoint_capture(')
        self.assertLess(capture.index('checkpoints.retire('),capture.index('checkpoint_transfer('))
        self.assertLess(capture.index('cudaStreamSynchronize'),capture.index('checkpoints.publish('))
        availability=block(current,'bool checkpoint_available(')
        availability=availability.replace(',bool image_mode=false', '').replace('        if(image_mode && (!image_checkpoints || !ss.image_points[(size_t)index].valid(p.pos))) return false;\n','').replace('!image_mode && !ss.imgs.empty()','!ss.imgs.empty()')
        self.assertEqual(availability,block(base,'bool checkpoint_available('),'disabled image mode retains original availability checks')
        marker='        const bool sampled_spec ='
        self.assertEqual(base[base.index(marker):base.index(';',base.index(marker))],current[current.index(marker):current.index(';',current.index(marker))])
        self.assertIn('if(!geni && fresh>1 && (!reuse || restored)) E.checkpoint_capture();',current)
        self.assertIn('if (!kv_pool_tokens || !kv_int8 || mtp_d != 2 || batch != 4 || !batch_mtp || vision ||',diagnostic)
        self.assertNotIn('cudaMalloc',(ROOT/'src/qwen35/q36_vision_staging.inc').read_text())

    def test_shared_host_syntax_not_actual_cuda_or_serve_compilation(self):
        run=subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic-errors','-fsyntax-only','-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/qwen35'),str(ROOT/'tests/core/q36_vision_syntax.cpp')],capture_output=True)
        self.assertEqual(run.returncode,0,run.stderr.decode())

if __name__=='__main__':unittest.main()
