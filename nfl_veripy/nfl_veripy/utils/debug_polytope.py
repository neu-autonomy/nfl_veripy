import numpy as np
import pypoman

B = np.array([[ 1.        ,  1.        ],
       [ 0.        ,  1.        ],
       [-0.93993515, -0.92576194],
       [ 0.1201297 , -0.8515239 ],
       [ 1.        ,  0.        ],
       [ 0.        ,  1.        ],
       [-1.        , -0.        ],
       [-0.        , -1.        ]])

c = np.array([ 5.5       ,  1.25      , -4.4964886 ,  0.25702265,  5.75      ,
        1.25      , -3.75      ,  1.25      ])

verts = pypoman.polygon.compute_polygon_hull(B, c)
import pdb; pdb.set_trace()