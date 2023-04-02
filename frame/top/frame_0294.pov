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
    ,<0.0002452750464606837,0.0,0.1005294066250862>,0.08
    ,<0.0004905307464218544,0.0,0.20103899555618548>,0.08
    ,<0.0007357378696381243,0.0,0.3015287746313346>,0.08
    ,<0.0009808964762210504,0.0,0.4019987516584879>,0.08
    ,<0.001226006369615153,0.0,0.5024489344411581>,0.08
    ,<0.0014710675402071676,0.0,0.6028793307781266>,0.08
    ,<0.0017160806061502758,0.0,0.7032899484629952>,0.08
    ,<0.0019610471404034897,0.0,0.803680795283942>,0.08
    ,<0.00220596979036902,0.0,0.9040518790235905>,0.08
    ,<0.002450852143146269,0.0,1.0044032074591587>,0.08
    ,<0.002695698332725391,0.0,1.104734788362799>,0.08
    ,<0.002940512448221011,0.0,1.205046629502099>,0.08
    ,<0.0031852978494745234,0.0,1.305338738640709>,0.08
    ,<0.0034300565251990855,0.0,1.4056111235388624>,0.08
    ,<0.003674788635037324,0.0,1.5058637919538327>,0.08
    ,<0.003919492355846284,0.0,1.6060967516400302>,0.08
    ,<0.0041641641120072695,0.0,1.7063100103488364>,0.08
    ,<0.0044087991917863886,0.0,1.8065035758280272>,0.08
    ,<0.004653392685558506,0.0,1.9066774558210253>,0.08
    ,<0.004897893465750528,0.0,2.006831658106957>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.004897870507648617,0.0,2.0068769466011833>,0.05
    ,<0.0048977003745261476,0.0,2.082092698858196>,0.05
    ,<0.0048975389900462575,0.0,2.157297356705186>,0.05
    ,<0.00489738635160497,0.0,2.232490923414542>,0.05
    ,<0.004897242456617802,0.0,2.307673402257203>,0.05
    ,<0.004897107302515879,0.0,2.382844796502677>,0.05
    ,<0.004896980886720462,0.0,2.4580051094190054>,0.05
    ,<0.004896863206637908,0.0,2.533154344272816>,0.05
    ,<0.004896754259681302,0.0,2.6082925043292633>,0.05
    ,<0.004896654043273056,0.0,2.6834195928520628>,0.05
    ,<0.004896562554831841,0.0,2.758535613103498>,0.05
    ,<0.004896479791774562,0.0,2.8336405683443964>,0.05
    ,<0.004896405751532387,0.0,2.9087344618341704>,0.05
    ,<0.004896340431544481,0.0,2.9838172968307717>,0.05
    ,<0.004896283829235039,0.0,3.0588890765907064>,0.05
    ,<0.004896235942019291,0.0,3.1339498043690672>,0.05
    ,<0.004896196767331421,0.0,3.2089994834194817>,0.05
    ,<0.004896166302621342,0.0,3.2840381169941626>,0.05
    ,<0.004896144545323849,0.0,3.3590657083438793>,0.05
    ,<0.004896131492866528,0.0,3.434082260717962>,0.05
    ,<0.004896127142687157,0.0,3.50908777736431>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }