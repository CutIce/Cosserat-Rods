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
    ,<5.3181998781487994e-05,0.0,0.10052939638099119>,0.08
    ,<0.00010635320216048741,0.0,0.20103897521915368>,0.08
    ,<0.000159515542539831,0.0,0.30152874433390736>,0.08
    ,<0.00021266979578602659,0.0,0.40199871154745953>,0.08
    ,<0.00026581688417297685,0.0,0.5024488847379438>,0.08
    ,<0.00031895787411701806,0.0,0.6028792717677993>,0.08
    ,<0.000372093925262451,0.0,0.7032898804717133>,0.08
    ,<0.00042522620352230406,0.0,0.8036807186656533>,0.08
    ,<0.00047835577568149584,0.0,0.9040517941436571>,0.08
    ,<0.0005314835098365176,0.0,1.0044031147024195>,0.08
    ,<0.0005846100049484418,0.0,1.104734688100221>,0.08
    ,<0.0006377355693753747,0.0,1.2050465220692905>,0.08
    ,<0.0006908602621661223,0.0,1.3053386243030454>,0.08
    ,<0.0007439839930451288,0.0,1.405611002524148>,0.08
    ,<0.0007971066678425138,0.0,1.5058636645249819>,0.08
    ,<0.0008502283604552047,0.0,1.6060966180686>,0.08
    ,<0.0009033494793551377,0.0,1.7063098708754616>,0.08
    ,<0.0009564708877909843,0.0,1.8065034307245609>,0.08
    ,<0.0010095939536944793,0.0,1.906677305411361>,0.08
    ,<0.001062712317101585,0.0,2.0068315026938133>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.001062689364829708,0.0,2.00687678980889>,0.05
    ,<0.0010625192752989818,0.0,2.0820925353976265>,0.05
    ,<0.0010623579329693236,0.0,2.1572971867576114>,0.05
    ,<0.0010622053353800975,0.0,2.2324907471807713>,0.05
    ,<0.0010620614800957845,0.0,2.3076732199846113>,0.05
    ,<0.001061926364433717,0.0,2.382844608474764>,0.05
    ,<0.0010617999855435534,0.0,2.458004915918216>,0.05
    ,<0.0010616823406424938,0.0,2.53315414557805>,0.05
    ,<0.0010615734268256153,0.0,2.6082923007564656>,0.05
    ,<0.0010614732410643077,0.0,2.6834193847700027>,0.05
    ,<0.0010613817804711102,0.0,2.758535400892195>,0.05
    ,<0.0010612990421428423,0.0,2.833640352356064>,0.05
    ,<0.001061225023310426,0.0,2.908734242411683>,0.05
    ,<0.0010611597213977426,0.0,2.983817074347854>,0.05
    ,<0.0010611031338710057,0.0,3.05888885145151>,0.05
    ,<0.0010610552583493898,0.0,3.1339495769754504>,0.05
    ,<0.0010610160925539802,0.0,3.2089992541589067>,0.05
    ,<0.0010609856341895728,0.0,3.2840378862591653>,0.05
    ,<0.0010609638810224085,0.0,3.3590654765478516>,0.05
    ,<0.001060950830941049,0.0,3.4340820282871998>,0.05
    ,<0.0010609464815684113,0.0,3.5090875447232017>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }