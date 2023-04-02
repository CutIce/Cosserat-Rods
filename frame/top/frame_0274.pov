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
    ,<0.0002285711966198687,0.0,0.10052940641678589>,0.08
    ,<0.00045711338006658713,0.0,0.20103899515961074>,0.08
    ,<0.0006856128889663133,0.0,0.3015287740521038>,0.08
    ,<0.0009140706839232332,0.0,0.4019987509014089>,0.08
    ,<0.0011424883811537035,0.0,0.5024489335095411>,0.08
    ,<0.0013708681286225715,0.0,0.602879329673459>,0.08
    ,<0.0015992122929034329,0.0,0.7032899471853167>,0.08
    ,<0.0018275230266366477,0.0,0.8036807938328728>,0.08
    ,<0.0020558018143983195,0.0,0.9040518773997815>,0.08
    ,<0.0022840491047057074,0.0,1.0044032056659833>,0.08
    ,<0.0025122641258680497,0.0,1.104734786407794>,0.08
    ,<0.0027404449517691074,0.0,1.2050466273978648>,0.08
    ,<0.002968588836159277,0.0,1.3053387364049471>,0.08
    ,<0.0031966927756623305,0.0,1.405611121193421>,0.08
    ,<0.0034247542221852896,0.0,1.5058637895226814>,0.08
    ,<0.0036527718104780007,0.0,1.6060967491466418>,0.08
    ,<0.003880745962316375,0.0,1.7063100078131241>,0.08
    ,<0.004108679219533407,0.0,1.8065035732637167>,0.08
    ,<0.004336576206282861,0.0,1.9066774532337174>,0.08
    ,<0.004564422053867188,0.0,2.006831655469714>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.004564399091132373,0.0,2.0068769439639556>,0.05
    ,<0.004564228923590405,0.0,2.0820926962210495>,0.05
    ,<0.004564067506311614,0.0,2.15729735406811>,0.05
    ,<0.004563914836739353,0.0,2.2324909207775248>,0.05
    ,<0.0045637709123128875,0.0,2.307673399620245>,0.05
    ,<0.0045636357304661545,0.0,2.3828447938657864>,0.05
    ,<0.004563509288627539,0.0,2.458005106782178>,0.05
    ,<0.004563391584219815,0.0,2.5331543416360405>,0.05
    ,<0.004563282614670999,0.0,2.6082925016925276>,0.05
    ,<0.004563182377418746,0.0,2.6834195902153697>,0.05
    ,<0.004563090869908533,0.0,2.758535610466849>,0.05
    ,<0.004563008089579731,0.0,2.833640565707784>,0.05
    ,<0.0045629340338570355,0.0,2.9087344591975937>,0.05
    ,<0.00456286870016956,0.0,2.9838172941942185>,0.05
    ,<0.0045628120859613755,0.0,3.0588890739541794>,0.05
    ,<0.0045627641886756614,0.0,3.1339498017325544>,0.05
    ,<0.0045627250057500325,0.0,3.208999480782992>,0.05
    ,<0.004562694534628871,0.0,3.2840381143576884>,0.05
    ,<0.004562672772748151,0.0,3.3590657057074127>,0.05
    ,<0.004562659717535158,0.0,3.4340822580815007>,0.05
    ,<0.004562655366434839,0.0,3.5090877747278575>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }