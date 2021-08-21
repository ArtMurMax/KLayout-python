import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point,DPoint,DSimplePolygon,SimplePolygon, DPolygon, Polygon,  Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from classLib.baseClasses import *
from classLib.coplanars import *

class BranchLine_finger(ComplexBase):
    def __init__( self, Z0, start, params, gndWidth = -1, trans_in=None ):
        if( len(params) != 6 ):
            raise LookupError
            return None
            
        self.params = params
        self.L1 = params[0]
        self.L2 = params[1]
        self.L3 = params[2]
        self.R1 = params[3]
        self.R2 = params[4]
        self.gamma = params[5]
        self.width = Z0.width
        self.gap = Z0.gap
        self.gndWidth = gndWidth
        self.dr = DPoint( 2*(self.L1 + self.L2*cos(self.gamma) + self.L3/2 + (self.R1 + self.R2)*sin(self.gamma)), 0 )
        self.L = 2*(self.L1 + self.L2 + self.L3/2 + (self.R1 + self.R2)*self.gamma)
        super( BranchLine_finger,self ).__init__( start, trans_in )
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

        
    def init_primitives( self ):
        self.connections = [DPoint(0,0),self.dr]
        self.angle_connections = [0,0]
        
        self.coplanar1 = CPW( self.width, self.gap, DPoint(0,0), DPoint(0,0) + DPoint(self.L1,0), self.gndWidth )
        self.arc1 = CPWArc(self.coplanar1, self.coplanar1.end, self.R1, self.gamma, self.gndWidth, Trans(Trans.M0))
        self.coplanar2 = CPW( self.width, self.gap, self.arc1.end, self.arc1.end + DPoint( cos(self.gamma),-sin(self.gamma) )*self.L2, self.gndWidth )
        self.arc2 = CPWArc(self.coplanar1, self.coplanar2.end, self.R2, self.gamma, self.gndWidth, DCplxTrans(1, -self.gamma * 180 / pi, False, 0, 0))
        self.coplanar3 = CPW( self.width, self.gap, self.arc2.end, self.arc2.end + DPoint( L3, 0 ), self.gndWidth )
        self.arc3 = CPWArc(self.coplanar1, self.coplanar3.end, self.R2, self.gamma, self.gndWidth)
        self.coplanar4 = CPW( self.width, self.gap, self.arc3.end, self.arc3.end + DPoint( cos(self.gamma),sin(self.gamma) )*self.L2, self.gndWidth )
        self.arc4 = CPWArc(self.coplanar1, self.coplanar4.end, self.R1, self.gamma, self.gndWidth, DCplxTrans(1, self.gamma * 180 / pi, True, 0, 0))
        self.coplanar5 = CPW( self.width, self.gap, self.arc4.end, self.arc4.end + DPoint(self.L1,0), self.gndWidth )
        self.primitives = {"coplanar1":self.coplanar1,"arc1":self.arc1,"coplanar2":self.coplanar2,"arc2":self.arc2,
                                "coplanar3":self.coplanar3, "arc3":self.arc3,"coplanar4":self.coplanar4,"arc4":self.arc4,
                                "coplanar5":self.coplanar5}

class BranchLine_finger2(ComplexBase):
    def __init__( self, Z0, start, params, gndWidth = -1, trans_in = None ):
        if( len(params) != 11 ):
            raise LookupError
            return None
            
        self.params = params
        self.L1 = params[0]
        self.L2 = params[1]
        self.L3 = params[2]
        self.L4 = params[3]
        self.L5 = params[4]
        self.R1 = params[5]
        self.R2 = params[6]
        self.R3 = params[7]
        self.R4 = params[8]
        self.gamma1 = params[9]
        self.gamma2 = params[10]
        self.width = Z0.width
        self.gap = Z0.gap
        self.gndWidth = gndWidth
        self.dr = DPoint( 2*(self.L1 + self.L2*cos(self.gamma1) + self.L3 + 
                                (self.R1 + self.R2)*sin(self.gamma1) + (self.R3 + self.R4)*sin(self.gamma2)+ 
                                + self.L4*cos(self.gamma2) + self.L5/2), 0 )
        self.L = 2*(self.L1 + self.L2 + self.L3 + (self.R1 + self.R2)*self.gamma1 + 
                        (self.R3 + self.R4)*self.gamma2 + self.L4 + self.L5/2)
        super( BranchLine_finger2,self ).__init__( start, trans_in )
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]
    
    def init_primitives( self ):
        self.connections = [DPoint(0,0),self.dr]
        self.angle_connections = [0,0]
        
        self.coplanar1 = CPW( self.width, self.gap, DPoint(0,0), DPoint(0,0) + DPoint(self.L1,0), self.gndWidth )
        self.arc1 = CPWArc(self.coplanar1, self.coplanar1.end, self.R1, self.gamma1, self.gndWidth, Trans(Trans.M0))
        self.coplanar2 = CPW( self.width, self.gap, self.arc1.end, self.arc1.end + DPoint( cos(self.gamma1),-sin(self.gamma1) )*self.L2, self.gndWidth )
        self.arc2 = CPWArc(self.coplanar1, self.coplanar2.end, self.R2, self.gamma1, self.gndWidth, DCplxTrans(1, -self.gamma1 * 180 / pi, False, 0, 0))
        self.coplanar3 = CPW( self.width, self.gap, self.arc2.end, self.arc2.end + DPoint( self.L3, 0 ), self.gndWidth )
        self.arc3 = CPWArc(self.coplanar1, self.coplanar3.end, self.R3, self.gamma2, self.gndWidth)
        self.coplanar4 = CPW( self.width, self.gap, self.arc3.end, self.arc3.end + DPoint( cos(self.gamma2),sin(self.gamma2) )*self.L4, self.gndWidth )
        self.arc4 = CPWArc(self.coplanar1, self.coplanar4.end, self.R4, self.gamma2, self.gndWidth, DCplxTrans(1, self.gamma2 * 180 / pi, True, 0, 0))
        self.coplanar5 = CPW( self.width, self.gap, self.arc4.end, self.arc4.end + DPoint(self.L5,0), self.gndWidth )
        self.arc5 = CPWArc(self.coplanar1, self.coplanar5.end, self.R4, self.gamma2, self.gndWidth, Trans(Trans.M0))
        self.coplanar6 = CPW( self.width, self.gap, self.arc5.end, self.arc5.end + DPoint( cos(self.gamma2),-sin(self.gamma2) )*self.L4, self.gndWidth )
        self.arc6 = CPWArc(self.coplanar1, self.coplanar6.end, self.R3, self.gamma2, self.gndWidth, DCplxTrans(1, -self.gamma2 * 180 / pi, False, 0, 0))
        self.coplanar7 = CPW( self.width, self.gap, self.arc6.end, self.arc6.end + DPoint( self.L3,0 ), self.gndWidth )
        self.arc7 = CPWArc(self.coplanar1, self.coplanar7.end, self.R2, self.gamma1, self.gndWidth)
        self.coplanar8 = CPW( self.width, self.gap, self.arc7.end, self.arc7.end + DPoint( cos(self.gamma1),sin(self.gamma1) )*self.L2, self.gndWidth )
        self.arc8 = CPWArc(self.coplanar1, self.coplanar8.end, self.R1, self.gamma1, self.gndWidth, DCplxTrans(1, self.gamma1 * 180 / pi, True, 0, 0))
        self.coplanar9 = CPW( self.width, self.gap, self.arc8.end, self.arc8.end + DPoint( self.L1,0 ), self.gndWidth )
        self.primitives = {"coplanar1":self.coplanar1,"coplanar2":self.coplanar2,"coplanar3":self.coplanar3,
                                "coplanar4":self.coplanar4,"coplanar5":self.coplanar5,"coplanar6":self.coplanar6,
                                "coplanar7":self.coplanar7,"coplanar8":self.coplanar8,"coplanar9":self.coplanar9,
                                "arc1":self.arc1,"arc2":self.arc2,"arc3":self.arc3,"arc4":self.arc4,
                                "arc5":self.arc5,"arc6":self.arc6,"arc7":self.arc7,"arc8":self.arc8}

class Coupler_BranchLine(ComplexBase):
    def __init__( self, origin, branchline_hor, branchline_vert, tJunction, trans_in=None ):
        self.bl_hor = branchline_hor
        self.bl_vert = branchline_vert
        self.tj = tJunction
        super( Coupler_BranchLine,self ).__init__( origin, trans_in )
        self.connections = [self.tj_BL.connections[0],self.tj_BR.connections[0],self.tj_TR.connections[0],self.tj_TL.connections[0]]
        self.angle_connections = [self.tj_BL.angle_connections[0],self.tj_BR.angle_connections[0],self.tj_TR.angle_connections[0],self.tj_TL.angle_connections[0]]
        self.dimensions = (self.metal_region + self.empty_region).bbox()
        
    def init_primitives( self ):
        Z0 = CPW( self.bl_vert.width, self.bl_vert.gap, DPoint(0,0), DPoint(0,0) )
        Z1 = CPW( self.bl_hor.width, self.bl_hor.gap, DPoint(0,0), DPoint(0,0) )
        self.tj_BL = TJunction_112( Z0,Z1, DPoint(0,0) + DPoint(0,Z0.b), Trans( Trans.M0 ) )
        self.bl_buttom = self.bl_hor.__class__( Z1, self.tj_BL.connections[2], self.bl_hor.params, trans_in=Trans(Trans.M0) )
        self.tj_BR = TJunction_112( Z0,Z1, self.bl_buttom.end + DPoint( Z0.b, Z1.b/2 ), Trans( Trans.R180 ) )
        self.bl_left = self.bl_vert.__class__( Z0, self.tj_BL.connections[1], self.bl_vert.params, trans_in=Trans(Trans.R90) )
        self.tj_TL = TJunction_112( Z0,Z1, self.bl_left.end + DPoint( -Z0.b/2,0 ) )
        self.bl_top = self.bl_hor.__class__( Z1, self.tj_TL.connections[2], self.bl_hor.params )
        self.tj_TR = TJunction_112( Z0,Z1, self.bl_top.end + DPoint( Z0.b,-Z1.b/2), Trans( Trans.R180*Trans.M0 ) )
        self.bl_right = self.bl_vert.__class__( Z0, self.tj_TR.connections[1], self.bl_vert.params, trans_in=Trans(Trans.R270) )
        self.primitives = {"tj_BL":self.tj_BL,"bl_buttom":self.bl_buttom,"tj_BR":self.tj_BR,"bl_left":self.bl_left,
                                "tj_TL":self.tj_TL,"bl_top":self.bl_top,"tj_TR":self.tj_TR,"bl_right":self.bl_right}
            
class TJunction_112( ElementBase ):
    def __init__( self, Z1, Z2, origin, trans_in=None ):
        self.Z1 = Z1
        self.Z2 = Z2
        super( TJunction_112,self ).__init__( origin, trans_in )
        
    def init_regions( self ):
        self.P0 = DPoint( 0,0 )
        self.P1 = self.P0 + DPoint( self.Z1.gap,0 )
        self.P2 = self.P1 + DPoint( self.Z1.width,0 )
        self.P3 = self.P2 + DPoint( 0,self.Z1.gap )
        self.P4 = self.P2 + DPoint( self.Z1.gap, self.Z2.gap )
        self.P5 = self.P4 + DPoint( 0,self.Z2.width )
        self.P6 = DPoint( 0, self.Z1.gap + self.Z1.width )
        self.P7 = DPoint( 0, self.Z1.gap )
        self.P8 = self.P7 + DPoint( self.Z1.gap,0 )
        self.P9 = self.P2 + DPoint( self.Z1.gap,0 )
        self.P10 = self.P5 + DPoint( 0, self.Z2.gap )
        self.P11 = self.P10 - DPoint( 2*self.Z1.gap + self.Z1.width, 0 )
        self.P12 = self.P6 + DPoint( 0, self.Z1.gap )
        
        self.connections = [(self.P6 + self.P7)*0.5,(self.P1 + self.P2)*0.5,(self.P5 + self.P4)*0.5]
        self.angle_connections = [0,3/2*pi,0]
                
        self.metal_polygon = DPolygon( [self.P1,self.P2,self.P3,self.P4,self.P5,self.P6,self.P7,self.P8] )
        self.empty1_polygon = DPolygon( [self.P0,self.P1,self.P8,self.P7] )
        self.empty2_polygon = DPolygon( [self.P2,self.P9,self.P4,self.P3]  )
        self.empty3_polygon = DPolygon( [self.P5,self.P10,self.P12,self.P6] )
        self.gnd_polygon = DPolygon( [self.P10,self.P11,self.P12] )
        self.metal_region = pya.Region( list(map(pya.Polygon().from_dpoly,[self.metal_polygon,self.gnd_polygon])) )
        self.empty_region = pya.Region( list(map(pya.Polygon().from_dpoly ,[self.empty1_polygon,self.empty2_polygon,self.empty3_polygon])) )