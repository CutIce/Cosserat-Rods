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
    ,<0.00019098802393457642,0.0,0.100529406001643>,0.08
    ,<0.00038196521878997513,0.0,0.20103899433967168>,0.08
    ,<0.000572905853835708,0.0,0.30152877284396146>,0.08
    ,<0.0007638106064992605,0.0,0.4019987493218464>,0.08
    ,<0.0009546795817233981,0.0,0.5024489315764591>,0.08
    ,<0.0011455124046587904,0.0,0.6028793274066205>,0.08
    ,<0.001336308478777336,0.0,0.7032899446067111>,0.08
    ,<0.0015270673558684803,0.0,0.8036807909663968>,0.08
    ,<0.0017177891344807611,0.0,0.9040518742703699>,0.08
    ,<0.00190847479207847,0.0,1.0044032022981366>,0.08
    ,<0.0020991263646928647,0.0,1.1047347828238898>,0.08
    ,<0.0022897469130771213,0.0,1.205046623616529>,0.08
    ,<0.002480340254503349,0.0,1.3053387324398587>,0.08
    ,<0.0026709104902297542,0.0,1.4056111170529164>,0.08
    ,<0.0028614613917141165,0.0,1.5058637852103693>,0.08
    ,<0.003051995759050646,0.0,1.6060967446629675>,0.08
    ,<0.0032425148813764564,0.0,1.7063100031579241>,0.08
    ,<0.0034330182135902058,0.0,1.8065035684391189>,0.08
    ,<0.0036235033703874804,0.0,1.9066774482471112>,0.08
    ,<0.003813987091875216,0.0,2.0068316503049326>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0038139641487565552,0.0,2.006876938799185>,0.05
    ,<0.0038137941266633856,0.0,2.082092691056354>,0.05
    ,<0.003813632847523313,0.0,2.1572973489034775>,0.05
    ,<0.0038134803087444486,0.0,2.23249091561295>,0.05
    ,<0.0038133365077197986,0.0,2.3076733944557275>,0.05
    ,<0.003813201441851967,0.0,2.3828447887013136>,0.05
    ,<0.0038130751085655836,0.0,2.4580051016177578>,0.05
    ,<0.003812957505293531,0.0,2.5331543364716587>,0.05
    ,<0.0038128486294586655,0.0,2.6082924965281884>,0.05
    ,<0.003812748478485367,0.0,2.6834195850510665>,0.05
    ,<0.0038126570497945774,0.0,2.7585356053025816>,0.05
    ,<0.003812574340793266,0.0,2.833640560543557>,0.05
    ,<0.003812500348898615,0.0,2.908734454033395>,0.05
    ,<0.003812435071547145,0.0,2.9838172890300436>,0.05
    ,<0.0038123785061734113,0.0,3.0588890687900285>,0.05
    ,<0.003812330650205263,0.0,3.133949796568426>,0.05
    ,<0.00381229150108766,0.0,3.208999475618878>,0.05
    ,<0.0038122610562636647,0.0,3.2840381091935784>,0.05
    ,<0.0038122393131550573,0.0,3.3590657005433173>,0.05
    ,<0.003812226269192803,0.0,3.434082252917414>,0.05
    ,<0.003812221921838809,0.0,3.509087769563767>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }