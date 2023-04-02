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
    ,<0.00019683394860054674,0.0,0.10052940606147938>,0.08
    ,<0.00039364470862814186,0.0,0.20103899446439474>,0.08
    ,<0.0005904168218948482,0.0,0.3015287730319855>,0.08
    ,<0.0007871504721703462,0.0,0.40199874957194615>,0.08
    ,<0.0009838458292588748,0.0,0.5024489318873696>,0.08
    ,<0.0011805034074746504,0.0,0.6028793277764999>,0.08
    ,<0.0013771243591595595,0.0,0.7032899450325122>,0.08
    ,<0.0015737106245460098,0.0,0.8036807914434339>,0.08
    ,<0.0017702648930055813,0.0,0.9040518747921683>,0.08
    ,<0.001966790357323503,0.0,1.004403202856666>,0.08
    ,<0.0021632902985293558,0.0,1.1047347834102403>,0.08
    ,<0.0023597675750702788,0.0,1.2050466242218778>,0.08
    ,<0.0025562241167383804,0.0,1.30533873305664>,0.08
    ,<0.0027526605338967403,0.0,1.4056111176759352>,0.08
    ,<0.0029490759402914187,0.0,1.50586378583762>,0.08
    ,<0.003145468055722881,0.0,1.60609674529597>,0.08
    ,<0.003341833613866761,0.0,1.7063100038013674>,0.08
    ,<0.0035381690260203945,0.0,1.806503569099861>,0.08
    ,<0.0037344712135622977,0.0,1.906677448932569>,0.08
    ,<0.003930689059003439,0.0,2.0068316510696396>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.003930666099311676,0.0,2.006876939563903>,0.05
    ,<0.003930495954385589,0.0,2.0820926918210976>,0.05
    ,<0.003930334558692168,0.0,2.157297349668255>,0.05
    ,<0.003930181909642656,0.0,2.232490916377768>,0.05
    ,<0.003930038004637174,0.0,2.307673395220575>,0.05
    ,<0.003929902841087079,0.0,2.382844789466184>,0.05
    ,<0.003929776416417783,0.0,2.458005102382654>,0.05
    ,<0.003929658728057025,0.0,2.5331543372365797>,0.05
    ,<0.003929549773434703,0.0,2.6082924972931267>,0.05
    ,<0.003929449549975695,0.0,2.683419585816031>,0.05
    ,<0.003929358055094731,0.0,2.758535606067558>,0.05
    ,<0.0039292752862049255,0.0,2.833640561308553>,0.05
    ,<0.0039292012407279005,0.0,2.9087344547984024>,0.05
    ,<0.003929135916093227,0.0,2.983817289795069>,0.05
    ,<0.003929079309734117,0.0,3.0588890695550552>,0.05
    ,<0.003929031419090268,0.0,3.133949797333462>,0.05
    ,<0.0039289922416055854,0.0,3.2089994763839225>,0.05
    ,<0.003928961774719242,0.0,3.284038109958627>,0.05
    ,<0.003928940015859596,0.0,3.3590657013083693>,0.05
    ,<0.00392892696245916,0.0,3.4340822536824662>,0.05
    ,<0.0039289226119639436,0.0,3.509087770328824>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }