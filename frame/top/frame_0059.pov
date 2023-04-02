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
    ,<4.900608754272426e-05,0.0,0.10052938330531119>,0.08
    ,<9.800272693500034e-05,0.0,0.20103894932723304>,0.08
    ,<0.00014699096191906678,0.0,0.3015287058781669>,0.08
    ,<0.00019597143524354186,0.0,0.401998660752264>,0.08
    ,<0.0002449448401896004,0.0,0.5024488218767373>,0.08
    ,<0.00029391199979145313,0.0,0.6028791969133539>,0.08
    ,<0.00034287391212564937,0.0,0.7032897935740845>,0.08
    ,<0.0003918317530163554,0.0,0.8036806197079431>,0.08
    ,<0.0004407868274505972,0.0,0.9040516832372374>,0.08
    ,<0.0004897404805989153,0.0,1.004402992104434>,0.08
    ,<0.0005386939785824116,0.0,1.1047345541579883>,0.08
    ,<0.0005876483856951419,0.0,1.2050463773635112>,0.08
    ,<0.0006366044681487381,0.0,1.3053384696856578>,0.08
    ,<0.0006855626439922083,0.0,1.4056108389996167>,0.08
    ,<0.0007345229984895272,0.0,1.505863493032387>,0.08
    ,<0.0007834853683332239,0.0,1.6060964394808945>,0.08
    ,<0.0008324494927064712,0.0,1.706309686120299>,0.08
    ,<0.0008814152026535221,0.0,1.806503240821323>,0.08
    ,<0.0009303826272250379,0.0,1.906677111427742>,0.08
    ,<0.0009793383803475269,0.0,2.0068313056500213>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0009793154326185535,0.0,2.0068765919505043>,0.05
    ,<0.0009791453760539933,0.0,2.0820923338009343>,0.05
    ,<0.0009789840627249249,0.0,2.1572969819134262>,0.05
    ,<0.000978831489562371,0.0,2.232490539590779>,0.05
    ,<0.0009786876537380047,0.0,2.3076730101460954>,0.05
    ,<0.000978552552807417,0.0,2.3828443968564894>,0.05
    ,<0.0009784261848516406,0.0,2.458004702955203>,0.05
    ,<0.0009783085478969101,0.0,2.533153931664021>,0.05
    ,<0.000978199639937433,0.0,2.608292086225292>,0.05
    ,<0.000978099459121106,0.0,2.6834191699051244>,0.05
    ,<0.0009780080029316068,0.0,2.758535185977207>,0.05
    ,<0.0009779252691291935,0.0,2.8336401377057028>,0.05
    ,<0.000977851256053712,0.0,2.90873402833075>,0.05
    ,<0.000977785961522923,0.0,2.983816861054856>,0.05
    ,<0.0009777293831650829,0.0,3.0588886390411254>,0.05
    ,<0.0009776815185545868,0.0,3.1339493654376476>,0.05
    ,<0.0009776423641630193,0.0,3.2089990434180535>,0.05
    ,<0.0009776119160653414,0.0,3.28403767620104>,0.05
    ,<0.0009775901709943577,0.0,3.3590652670265437>,0.05
    ,<0.00097757712593579,0.0,3.434081819117176>,0.05
    ,<0.00097757277819248,0.0,3.509087335676693>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }