#include "E:/Documents/23SP/graduation_project/PyElastica-master/examples/Visualization/default.inc"

camera{
    location <0,15,3>
    angle 30
    look_at <0.0,0,3>
    sky <-1,0,0>
    right x*image_width/image_height
}
light_source{
    <15,10.5,-15>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<0.0,0.0,0.0>,0.08
    ,<4.984130164687916e-05,0.0,0.10052939768438374>,0.08
    ,<9.967268391376856e-05,0.0,0.20103897770496468>,0.08
    ,<0.00014949554718822624,0.0,0.3015287477895103>,0.08
    ,<0.00019931055121747598,0.0,0.40199871583649815>,0.08
    ,<0.0002491184346172149,0.0,0.502448889594472>,0.08
    ,<0.00029892008112628125,0.0,0.6028792767553465>,0.08
    ,<0.0003487165502052808,0.0,0.7032898850775433>,0.08
    ,<0.00039850905667429646,0.0,0.8036807222478577>,0.08
    ,<0.0004482989047020788,0.0,0.9040517960794078>,0.08
    ,<0.0004980873857782534,0.0,1.0044031143738823>,0.08
    ,<0.0005478756594054638,0.0,1.104734684908324>,0.08
    ,<0.0005976646430351471,0.0,1.2050465156097894>,0.08
    ,<0.0006474549360680567,0.0,1.305338614456983>,0.08
    ,<0.0006972467999886957,0.0,1.4056109892677577>,0.08
    ,<0.000747040206511824,0.0,1.5058636478964798>,0.08
    ,<0.0007968349525224747,0.0,1.606096598269923>,0.08
    ,<0.0008466308271525224,0.0,1.7063098481635786>,0.08
    ,<0.0008964278087155744,0.0,1.8065034053484363>,0.08
    ,<0.000946226252270516,0.0,1.906677277671298>,0.08
    ,<0.0009960129820325543,0.0,2.0068314728812164>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.000995990033521589,0.0,2.0068767593750434>,0.05
    ,<0.0009958199718182316,0.0,2.0820925019653007>,0.05
    ,<0.0009956586562933222,0.0,2.157297150416048>,0.05
    ,<0.00099550608454848,0.0,2.2324907080484206>,0.05
    ,<0.0009953622539213647,0.0,2.307673178182979>,0.05
    ,<0.0009952271615860458,0.0,2.3828445641129576>,0.05
    ,<0.0009951008048908465,0.0,2.4580048691283904>,0.05
    ,<0.0009949831809639269,0.0,2.5331540965375496>,0.05
    ,<0.000994874286681256,0.0,2.6082922496524588>,0.05
    ,<0.0009947741188879155,0.0,2.6834193317690405>,0.05
    ,<0.0009946826747269733,0.0,2.758535346167142>,0.05
    ,<0.0009945999513186182,0.0,2.8336402961108647>,0.05
    ,<0.0009945259455240386,0.0,2.9087341848399464>,0.05
    ,<0.0009944606546108045,0.0,2.9838170155790746>,0.05
    ,<0.0009944040762015318,0.0,3.058888791570456>,0.05
    ,<0.0009943562078109829,0.0,3.1339495160888844>,0.05
    ,<0.0009943170471603897,0.0,3.2089991924170906>,0.05
    ,<0.0009942865923683148,0.0,3.284037823820757>,0.05
    ,<0.000994264841471732,0.0,3.3590654135629614>,0.05
    ,<0.000994251792431416,0.0,3.4340819649326453>,0.05
    ,<0.0009942474432978955,0.0,3.509087481234873>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }