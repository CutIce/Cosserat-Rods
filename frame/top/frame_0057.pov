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
    ,<4.7335662421459435e-05,0.0,0.10052942790138046>,0.08
    ,<9.466300067789925e-05,0.0,0.2010390380769744>,0.08
    ,<0.00014198219966541687,0.0,0.30152883854092905>,0.08
    ,<0.00018929388782003544,0.0,0.4019988371282316>,0.08
    ,<0.00023659868474063025,0.0,0.5024490415563496>,0.08
    ,<0.0002838972905603171,0.0,0.6028794597906253>,0.08
    ,<0.0003311905615170076,0.0,0.7032900996642485>,0.08
    ,<0.0003784795498208356,0.0,0.8036809688717483>,0.08
    ,<0.00042576549776477703,0.0,0.9040520752623032>,0.08
    ,<0.00047304978129577793,0.0,1.004403426659288>,0.08
    ,<0.0005203338072573484,0.0,1.1047350306620491>,0.08
    ,<0.0005676188857030938,0.0,1.205046894802735>,0.08
    ,<0.0006149061025172138,0.0,1.3053390266353564>,0.08
    ,<0.0006621962179700034,0.0,1.4056114338099457>,0.08
    ,<0.0007094896171350878,0.0,1.5058641239251782>,0.08
    ,<0.0007567863295613634,0.0,1.6060971044362935>,0.08
    ,<0.0008040861269256091,0.0,1.7063103829351367>,0.08
    ,<0.0008513886846846244,0.0,1.8065039670923937>,0.08
    ,<0.0008986937874462193,0.0,1.9066778646550675>,0.08
    ,<0.0009459894542754491,0.0,2.0068320832474487>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0009459664951874922,0.0,2.006877376768719>,0.05
    ,<0.0009457963540479417,0.0,2.082093153257486>,0.05
    ,<0.0009456349597475268,0.0,2.1572978344943747>,0.05
    ,<0.0009454823097186502,0.0,2.2324914237277373>,0.05
    ,<0.0009453384015124244,0.0,2.3076739241893907>,0.05
    ,<0.000945203232802822,0.0,2.382845339103255>,0.05
    ,<0.0009450768020398364,0.0,2.458005671664032>,0.05
    ,<0.0009449591073079077,0.0,2.5331549250158427>,0.05
    ,<0.0009448501463099515,0.0,2.6082931023136755>,0.05
    ,<0.0009447499171148436,0.0,2.6834202068054354>,0.05
    ,<0.0009446584175096914,0.0,2.7585362417930015>,0.05
    ,<0.0009445756452148008,0.0,2.8336412105184206>,0.05
    ,<0.0009445015978807222,0.0,2.90873511616393>,0.05
    ,<0.0009444362729814803,0.0,2.983817961967442>,0.05
    ,<0.0009443796683478348,0.0,3.0588897512470545>,0.05
    ,<0.0009443317810517335,0.0,3.133950487277553>,0.05
    ,<0.0009442926076848575,0.0,3.209000173217139>,0.05
    ,<0.0009442621452817438,0.0,3.2840388121971174>,0.05
    ,<0.0009442403905347887,0.0,3.3590664074209937>,0.05
    ,<0.0009442273397422514,0.0,3.434082962135408>,0.05
    ,<0.0009442229900317381,0.0,3.5090884795678208>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }