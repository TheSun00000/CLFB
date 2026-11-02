#!/usr/bin/env python3
"""merge_ckpt.py <out.pickle> <fa.pickle> [<b.pickle>]: checkpoint with F and the actor of <fa> and the backward network of <b> (default: that of <fa>).
Used by the stream chains of tools/queue_stream.py: B comes from the stage k of a stream, F and the actor from the previous F / actor stage. Needs no GPU."""
import pickle, sys
out, fa = sys.argv[1], sys.argv[2]
a = pickle.load(open(fa, 'rb'))
b = pickle.load(open(sys.argv[3], 'rb')) if len(sys.argv) > 3 else a
fb = dict(a['fb_params']); fb['backward'] = b['fb_params']['backward']
ck = {'fb_params': fb, 'actor_params': a['actor_params']}
pickle.dump(ck, open(out, 'wb'))
print('merged', out, 'F+actor from', fa, 'B from', sys.argv[3] if len(sys.argv) > 3 else fa)
