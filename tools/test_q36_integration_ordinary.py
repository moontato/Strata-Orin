import unittest
import q36_adaptive_integration_fixture as fixture
def encode(messages):
    return list(''.join(m['role']+':'+m['content']+'\n' for m in messages).encode())
def verify(generate):
    _, _, audit, cases = generate(encode)
    rows = cases['ordinary']
    assert len(rows) == 4 and all(ms[-1]['role']=='user' for _,ms in rows)
    assert [r['classification'] for r in audit['turns'] if r['case']=='ordinary']==['new','append','append','append']
    assert all(b[:len(a)]==a for (_,a),(_,b) in zip(rows,rows[1:]))
class Ordinary(unittest.TestCase):
    def test_user_ending_continuations(self): verify(fixture.generate)
if __name__=='__main__': unittest.main()
