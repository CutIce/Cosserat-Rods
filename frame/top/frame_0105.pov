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
    ,<8.742479599497257e-05,0.0,0.10052940524379746>,0.08
    ,<0.00017484078370505227,0.0,0.2010389928679577>,0.08
    ,<0.0002622414882870139,0.0,0.3015287706866327>,0.08
    ,<0.0003496278808911431,0.0,0.40199874650698875>,0.08
    ,<0.00043700076034413043,0.0,0.5024489281316262>,0.08
    ,<0.0005243606526803176,0.0,0.6028793233586801>,0.08
    ,<0.0006117077807775274,0.0,0.7032899399816692>,0.08
    ,<0.0006990421235822454,0.0,0.8036807857895651>,0.08
    ,<0.0007863635550050456,0.0,0.90405186856683>,0.08
    ,<0.0008736720358675457,0.0,1.0044031960933013>,0.08
    ,<0.0009609678210018456,0.0,1.1047347761441542>,0.08
    ,<0.001048251635490908,0.0,1.2050466164897975>,0.08
    ,<0.0011355247769324784,0.0,1.3053387248958843>,0.08
    ,<0.0012227891113925464,0.0,1.4056111091232932>,0.08
    ,<0.001310046947956832,0.0,1.505863776928242>,0.08
    ,<0.0013973008016601358,0.0,1.6060967360622906>,0.08
    ,<0.0014845530839842072,0.0,1.7063099942723925>,0.08
    ,<0.001571805757555215,0.0,1.8065035593011627>,0.08
    ,<0.0016590600292556473,0.0,1.9066774388868406>,0.08
    ,<0.001746340546530419,0.0,2.0068316407558093>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0017463176023136144,0.0,2.006876929250294>,0.05
    ,<0.001746147572052429,0.0,2.0820926815085614>,0.05
    ,<0.001745986285078024,0.0,2.1572973393566697>,0.05
    ,<0.0017458337388209283,0.0,2.232490906067027>,0.05
    ,<0.001745689930712163,0.0,2.307673384910598>,0.05
    ,<0.0017455548581700663,0.0,2.382844779156881>,0.05
    ,<0.001745428518617883,0.0,2.4580050920739187>,0.05
    ,<0.0017453109094847253,0.0,2.533154326928349>,0.05
    ,<0.0017452020281887772,0.0,2.6082924869853232>,0.05
    ,<0.0017451018721493926,0.0,2.683419575508568>,0.05
    ,<0.0017450104388033807,0.0,2.758535595760392>,0.05
    ,<0.0017449277255896607,0.0,2.8336405510016474>,0.05
    ,<0.0017448537299356221,0.0,2.9087344444917207>,0.05
    ,<0.0017447884492682313,0.0,2.9838172794885587>,0.05
    ,<0.0017447318810226307,0.0,3.0588890592486697>,0.05
    ,<0.001744684022637201,0.0,3.133949787027173>,0.05
    ,<0.0017446448715469735,0.0,3.208999466077699>,0.05
    ,<0.001744614425185203,0.0,3.2840380996524554>,0.05
    ,<0.0017445926809873781,0.0,3.359065691002205>,0.05
    ,<0.0017445796363902426,0.0,3.4340822433762757>,0.05
    ,<0.0017445752888313548,0.0,3.5090877600226222>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }