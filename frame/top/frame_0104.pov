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
    ,<8.658960348250315e-05,0.0,0.10052940522939323>,0.08
    ,<0.00017316993633776411,0.0,0.2010389928393804>,0.08
    ,<0.0002597352522018657,0.0,0.3015287706438609>,0.08
    ,<0.0003462865549663525,0.0,0.40199874645004463>,0.08
    ,<0.0004328247230081186,0.0,0.5024489280606369>,0.08
    ,<0.0005193503862854671,0.0,0.6028793232737801>,0.08
    ,<0.0006058638676264033,0.0,0.7032899398830458>,0.08
    ,<0.000692365202198352,0.0,0.8036807856774573>,0.08
    ,<0.000778854246938507,0.0,0.9040518684413987>,0.08
    ,<0.0008653308551586539,0.0,1.004403195954732>,0.08
    ,<0.0009517950847618433,0.0,1.1047347759927704>,0.08
    ,<0.0010382473969471329,0.0,1.205046616326181>,0.08
    ,<0.0011246887995400062,0.0,1.3053387247207378>,0.08
    ,<0.0012111208964797457,0.0,1.4056111089374106>,0.08
    ,<0.0012975458195662285,0.0,1.5058637767324936>,0.08
    ,<0.0013839660375964795,0.0,1.606096735857607>,0.08
    ,<0.0014703840759265762,0.0,1.706309994059716>,0.08
    ,<0.0015568021726744537,0.0,1.806503559081263>,0.08
    ,<0.001643221939799237,0.0,1.906677438660393>,0.08
    ,<0.001729668718762122,0.0,2.006831640523412>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0017296457740791414,0.0,2.0068769290162765>,0.05
    ,<0.0017294757403648862,0.0,2.082092681266694>,0.05
    ,<0.0017293144501236658,0.0,2.1572973391072114>,0.05
    ,<0.0017291619007779222,0.0,2.2324909058102436>,0.05
    ,<0.0017290180897562722,0.0,2.307673384646835>,0.05
    ,<0.0017288830144841184,0.0,2.3828447788865317>,0.05
    ,<0.0017287566723843564,0.0,2.4580050917974865>,0.05
    ,<0.0017286390608798655,0.0,2.533154326646294>,0.05
    ,<0.0017285301773960215,0.0,2.608292486698174>,0.05
    ,<0.0017284300193591605,0.0,2.6834195752168712>,0.05
    ,<0.0017283385841935282,0.0,2.758535595464697>,0.05
    ,<0.0017282558693209877,0.0,2.8336405507024214>,0.05
    ,<0.0017281818721635186,0.0,2.9087344441894434>,0.05
    ,<0.001728116590149845,0.0,2.9838172791837296>,0.05
    ,<0.0017280600207186625,0.0,3.058889058941762>,0.05
    ,<0.0017280121613131522,0.0,3.1339497867185755>,0.05
    ,<0.0017279730093775418,0.0,3.2089994657676977>,0.05
    ,<0.0017279425623584993,0.0,3.2840380993413727>,0.05
    ,<0.0017279208177012383,0.0,3.3590656906903495>,0.05
    ,<0.0017279077728389893,0.0,3.434082243063998>,0.05
    ,<0.0017279034251945667,0.0,3.50908775971022>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }