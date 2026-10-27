import tempfile
from pathlib import Path
import unittest
from tools import q36_adaptive_checkpoint_fixture as f


class Fixtures(unittest.TestCase):
    def test_confirmed_sequence_character_lengths_not_token_hints(self):
        a, b, c = f.cases()['combined'][:3]
        system_a, user_a = a['messages'][:2]
        system_b, user_b = b['messages'][:2]
        system_c, user_c = c['messages'][:2]
        self.assertEqual(system_a, system_b)
        s1, s2 = system_a['content'], system_c['content']
        self.assertEqual((len(s1), len(s2)), (16852, 16852))
        self.assertEqual(next(i for i, (x, y) in enumerate(zip(s1, s2)) if x != y), 13917)
        self.assertEqual((len(user_a['content']), len(user_b['content'])), (52, 2459))
        self.assertIn(user_a['content'], user_b['content'])
        self.assertEqual(user_a, user_c)
        self.assertTrue(all(m in c['messages'] for m in b['messages'][2:5]))
        a2, b2, c2 = f.cases()['user_zero_offset'][:3]
        self.assertEqual((len(a2['messages'][1]['content']), len(b2['messages'][1]['content'])), (128, 2635))
        self.assertNotEqual(a2['messages'][1]['content'][0], b2['messages'][1]['content'][0])
        self.assertEqual(a2, c2)

    def test_encode_uses_rendered_ids_and_writes_all_sequences(self):
        # Deliberately compress pairs of characters: character offsets cannot
        # accidentally masquerade as rendered-token offsets in metrics.
        def encode(item):
            rendered = '|'.join(m['role'] + ':' + m['content'] for m in item['messages'])
            return [sum(map(ord, rendered[i:i+2])) for i in range(0, len(rendered), 2)]
        encoded = f.encode_cases(encode)
        with tempfile.TemporaryDirectory() as d:
            metrics = f.write_fixture(d, encoded)
            self.assertEqual(len(list(Path(d).glob('*.csv'))), 7)
            self.assertNotEqual(metrics['system'][1]['common_tokens'], 13917)
            self.assertEqual(metrics['append'][1]['append_only'], True)
            for name, rows in metrics.items():
                self.assertEqual(len(rows), len(encoded[name]))
                self.assertEqual(rows[0]['common_tokens'], 0)
                for row in rows:
                    self.assertLessEqual(row['common_tokens'], row['tokens'])


if __name__ == '__main__':
    unittest.main()
