"""Real CPU decoding, malformed-image control, and preservation of HTTP errors."""
from pathlib import Path
import ast
import hashlib
import io
import json
import sys
import time
import urllib.error
import zlib

import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import q36_image_http_smoke as h


@pytest.mark.parametrize('rgb',[(200,40,40),(40,60,200)])
def test_generated_png_decodes_in_an_independent_library(tmp_path,rgb):
    p=h.png(tmp_path/'fixture.png',rgb)
    with Image.open(p) as im:
        im.load()
        assert im.size==(64,64) and im.mode=='RGB'
        assert set(im.getdata())=={rgb}
    assert int.from_bytes(p.read_bytes()[8:12],'big')==13


def test_original_bad_length_cannot_pass_preflight(tmp_path):
    # Reproduce precisely the rejected image, including its recorded hash.
    def chunk(tag,data):
        body=tag+data
        return len(body).to_bytes(4,'big')+body+zlib.crc32(body).to_bytes(4,'big')
    raw=b''.join(b'\0'+bytes((200,40,40))*64 for _ in range(64))
    data=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',(64).to_bytes(4,'big')*2+b'\x08\x02\0\0\0')
          +chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
    assert hashlib.sha256(data).hexdigest()=='a583d29ea259af46702a2b3f8f5e90f12bfacea8e8b624e10017891bf258ebdb'
    with pytest.raises(ValueError,match='PNG fixture'):
        h.validate_generated_png(data,(200,40,40))
    with pytest.raises(Exception):
        Image.open(io.BytesIO(data)).load()


def test_regressed_generator_fails_before_writing_an_image(tmp_path):
    source=Path(h.__file__).read_text()
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('png','validate_generated_png')]
    code='\n\n'.join(ast.get_source_segment(source,n) for n in nodes)
    assert code.count('len(data).to_bytes(4,\'big\')')==1
    code=code.replace('len(data).to_bytes(4,\'big\')','len(body).to_bytes(4,\'big\')')
    ns=dict(Path=Path,zlib=zlib)
    exec(compile(code,'bad-length-mutation','exec'),ns)
    with pytest.raises(ValueError):ns['png'](tmp_path/'bad.png',(200,40,40))
    assert not (tmp_path/'bad.png').exists()


def test_corrupt_crc_is_rejected(tmp_path):
    data=bytearray(h.png(tmp_path/'a.png',(200,40,40)).read_bytes())
    data[30]^=1
    with pytest.raises(ValueError):h.validate_generated_png(bytes(data),(200,40,40))


def test_http_400_body_and_failed_request_are_preserved(tmp_path):
    body=b'{"error":{"message":"specific encoder rejection"}}'
    class Client:
        def _json(self,*args):
            raise urllib.error.HTTPError('http://127.0.0.1/v1/chat/completions',400,'Bad Request',{},io.BytesIO(body))
    step=dict(step='stable_image/1',conv='stable',op='user',text='public test',image='A')
    image=h.png(tmp_path/'a.png',(200,40,40))
    with pytest.raises(RuntimeError,match='specific encoder rejection'):
        h.run_steps(Client(),'test',[step],24,{'A':image},'marker',time.monotonic()+3,'baseline',out=tmp_path)
    assert (tmp_path/'baseline-stable_image-1-request.json').is_file()
    error=json.loads((tmp_path/'baseline-stable_image-1-error.json').read_text())
    assert error['status']==400 and error['body']==body.decode()
