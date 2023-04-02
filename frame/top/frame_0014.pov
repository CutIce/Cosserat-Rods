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
    ,<1.1344530605830051e-05,0.0,0.10051337456319903>,0.08
    ,<2.2688679087363467e-05,0.0,0.20100773964924049>,0.08
    ,<3.4031921934493625e-05,0.0,0.30148306350780507>,0.08
    ,<4.537586863598654e-05,0.0,0.4019392168688227>,0.08
    ,<5.672218693859404e-05,0.0,0.502376137531044>,0.08
    ,<6.807231550958362e-05,0.0,0.6027937308962944>,0.08
    ,<7.942709711528914e-05,0.0,0.7031918767767639>,0.08
    ,<9.078730480729706e-05,0.0,0.8035706615986576>,0.08
    ,<0.00010215384017285198,0.0,0.9039301155273085>,0.08
    ,<0.00011352795957305715,0.0,1.004270015155022>,0.08
    ,<0.00012491159308257126,0.0,1.1045902056001489>,0.08
    ,<0.00013630550047263952,0.0,1.2048906787069353>,0.08
    ,<0.0001477105049841852,0.0,1.3051713796381763>,0.08
    ,<0.00015912810563440882,0.0,1.40543244631277>,0.08
    ,<0.00017055813237456114,0.0,1.5056739651715532>,0.08
    ,<0.0001820005060061599,0.0,1.6058961000524037>,0.08
    ,<0.00019345637901153892,0.0,1.70609907301019>,0.08
    ,<0.00020492590380786013,0.0,1.806282786349422>,0.08
    ,<0.00021641049141319374,0.0,1.9064471265451084>,0.08
    ,<0.00022791419905304002,0.0,2.0065918835669936>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0002278860205442792,0.0,2.006634202100212>,0.05
    ,<0.0002276767874218988,0.0,2.0818353756198658>,0.05
    ,<0.00022747804799656804,0.0,2.1570253562971016>,0.05
    ,<0.00022728959767428827,0.0,2.2322041245790354>,0.05
    ,<0.0002271113057632845,0.0,2.307371677528706>,0.05
    ,<0.00022694348504247655,0.0,2.3825280682197727>,0.05
    ,<0.0002267860773127516,0.0,2.45767344532469>,0.05
    ,<0.00022663916415613418,0.0,2.5328080213082904>,0.05
    ,<0.00022650334440561936,0.0,2.607931963175618>,0.05
    ,<0.00022637852636180043,0.0,2.6830453155497165>,0.05
    ,<0.00022626436545431799,0.0,2.7581480633832163>,0.05
    ,<0.0002261614434788237,0.0,2.833240263834605>,0.05
    ,<0.00022607019910701192,0.0,2.9083220741095652>,0.05
    ,<0.00022599045043285098,0.0,2.983393667331634>,0.05
    ,<0.00022592177941379627,0.0,3.0584551969246>,0.05
    ,<0.0002258640918373043,0.0,3.13350683908315>,0.05
    ,<0.0002258168643935714,0.0,3.2085487701341493>,0.05
    ,<0.00022577939956008674,0.0,3.283581084310325>,0.05
    ,<0.00022575220043274353,0.0,3.3586038472145328>,0.05
    ,<0.00022573578629754768,0.0,3.43361726830931>,0.05
    ,<0.00022573027499171776,0.0,3.5086216837731743>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }