from maps import *
from simple_polygon import Simple_Polygon
from utilities import mk_regpoly, mk_obstacle, mk_spiky_circle, mk_spiky_obstacle

# some simple environments for experiments
 
def square(L):
    return Simple_Polygon("square",np.array([[0.0,0.0], [L, 0.0],[L,L],[0.0,L]]))

def octagon(L):
    return Simple_Polygon("octagon",np.array(mk_regpoly(8, L)))

def spikes(L):
    return Simple_Polygon("spikes",np.array(mk_spiky_circle(8, 0.5*L)))


def spike_annulus(L):
    return Simple_Polygon("spk_ring",
                           np.array(mk_spiky_circle(8, 0.5*L)),
                           [np.array(mk_obstacle(mk_regpoly(4, 0.4*L)))])

square_hole = Simple_Polygon("sqh",simple_holes[0], simple_holes[1])
l_poly = [np.array([(0, 0), (60, 0), (60, 23), (26, 23), (26, 46), (0, 46)],
dtype=np.float)]

def shelfp():
    return Simple_Polygon("shelf",shelf[0])
# octagon divided into five regions

def simple_nonconv_p():
    return Simple_Polygon("simple_nonconv",simple_nonconv[0])

def octagon_rs(L):
    oct_verts = np.array(mk_regpoly(8, 0.8*L, offset=np.pi/8.))
    octagon = Simple_Polygon("octagon", oct_verts)

    wire_verts = np.array(mk_regpoly(4, 0.4*L, offset=np.pi/4))
    r1 = [wire_verts[0], midpoint(oct_verts[0], oct_verts[1]), oct_verts[1], oct_verts[2],
    midpoint(oct_verts[2], oct_verts[3]), wire_verts[1]]
    r2 = [wire_verts[1], midpoint(oct_verts[2], oct_verts[3]), oct_verts[3], oct_verts[4],
    midpoint(oct_verts[4], oct_verts[5]), wire_verts[2]]
    r3 = [wire_verts[2], midpoint(oct_verts[4], oct_verts[5]), oct_verts[5], oct_verts[6],
    midpoint(oct_verts[6], oct_verts[7]), wire_verts[3]]
    r4 = [wire_verts[3], midpoint(oct_verts[6], oct_verts[7]), oct_verts[7], oct_verts[0],
    midpoint(oct_verts[0], oct_verts[1]), wire_verts[0]]

    rs = [wire_verts, r1, r2, r3, r4]
    rs_as_obs = [mk_obstacle(r) for r in rs]
    regions = [Simple_Polygon("r"+str(i), np.array(vs)) for i,vs in enumerate(rs)]
    return regions

