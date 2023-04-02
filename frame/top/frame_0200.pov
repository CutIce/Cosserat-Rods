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
    ,<0.00016676750104717053,0.0,0.10052940577367013>,0.08
    ,<0.00033352294866966474,0.0,0.20103899389809765>,0.08
    ,<0.0005002464604268162,0.0,0.30152877219740676>,0.08
    ,<0.000666938640730639,0.0,0.401998748478959>,0.08
    ,<0.0008335996075134666,0.0,0.5024489305458063>,0.08
    ,<0.0010002291226623246,0.0,0.6028793261965903>,0.08
    ,<0.001166826853341039,0.0,0.7032899432253874>,0.08
    ,<0.0013333927070896111,0.0,0.80368078942156>,0.08
    ,<0.0014999271630775745,0.0,0.9040518725694962>,0.08
    ,<0.001666431517514142,0.0,1.0044032004485213>,0.08
    ,<0.0018329079754554117,0.0,1.1047347808328247>,0.08
    ,<0.0019993595466130175,0.0,1.2050466214915339>,0.08
    ,<0.0021657897478632015,0.0,1.3053387301888764>,0.08
    ,<0.0023322021436117135,0.0,1.4056111146844843>,0.08
    ,<0.002498599801335671,0.0,1.505863782733692>,0.08
    ,<0.0026649847635317964,0.0,1.6060967420878716>,0.08
    ,<0.0028313576502516456,0.0,1.7063100004946612>,0.08
    ,<0.0029977174820729493,0.0,1.8065035656980695>,0.08
    ,<0.0031640617964151054,0.0,1.9066774454384396>,0.08
    ,<0.0033303973396407874,0.0,2.0068316474460888>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0033303743946675518,0.0,2.0068769359403626>,0.05
    ,<0.0033302043588372913,0.0,2.0820926881976147>,0.05
    ,<0.0033300430666519643,0.0,2.1572973460448237>,0.05
    ,<0.003329890515527197,0.0,2.232490912754393>,0.05
    ,<0.0033297467028774773,0.0,2.307673391597258>,0.05
    ,<0.0033296116261119484,0.0,2.3828447858429165>,0.05
    ,<0.0033294852826420093,0.0,2.4580050987594286>,0.05
    ,<0.0033293676698889618,0.0,2.5331543336133975>,0.05
    ,<0.0033292587852754925,0.0,2.608292493669986>,0.05
    ,<0.0033291586262149267,0.0,2.683419582192927>,0.05
    ,<0.003329067190125378,0.0,2.758535602444488>,0.05
    ,<0.0033289844744446564,0.0,2.833640557685514>,0.05
    ,<0.003328910476607377,0.0,2.908734451175392>,0.05
    ,<0.0033288451940274477,0.0,2.983817286172081>,0.05
    ,<0.0033287886241285015,0.0,3.058889065932096>,0.05
    ,<0.003328740764349315,0.0,3.133949793710515>,0.05
    ,<0.0033287016121199766,0.0,3.2089994727609894>,0.05
    ,<0.0033286711648607774,0.0,3.284038106335717>,0.05
    ,<0.0033286494200071817,0.0,3.359065697685463>,0.05
    ,<0.003328636375012771,0.0,3.4340822500595594>,0.05
    ,<0.003328632027322025,0.0,3.509087766705917>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }