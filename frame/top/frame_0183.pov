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
    ,<0.00015256900280498993,0.0,0.10052940565448804>,0.08
    ,<0.00030511245093254636,0.0,0.20103899367406225>,0.08
    ,<0.0004576275287814448,0.0,0.30152877187255195>,0.08
    ,<0.0006101149805604001,0.0,0.401998748057244>,0.08
    ,<0.0007625759826918804,0.0,0.5024489300305681>,0.08
    ,<0.0009150121526527583,0.0,0.6028793255901043>,0.08
    ,<0.0010674254152579891,0.0,0.7032899425286546>,0.08
    ,<0.0012198177535174203,0.0,0.8036807886343961>,0.08
    ,<0.0013721908963429934,0.0,0.9040518716910271>,0.08
    ,<0.0015245460116618797,0.0,1.0044031994779647>,0.08
    ,<0.0016768834786147103,0.0,1.1047347797704532>,0.08
    ,<0.0018292028012603248,0.0,1.2050466203396193>,0.08
    ,<0.001981502702260297,0.0,1.3053387289524347>,0.08
    ,<0.0021337814005467732,0.0,1.4056111133715539>,0.08
    ,<0.002286037035816816,0.0,1.5058637813550988>,0.08
    ,<0.002438268179827808,0.0,1.6060967406563773>,0.08
    ,<0.002590474341572505,0.0,1.7063099990236228>,0.08
    ,<0.002742656357355631,0.0,1.80650356419977>,0.08
    ,<0.00289481657750852,0.0,1.9066774439223835>,0.08
    ,<0.0030469322219998955,0.0,2.006831645938072>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0030469092623276197,0.0,2.006876934432365>,0.05
    ,<0.003046739117501824,0.0,2.082092686689712>,0.05
    ,<0.0030465777218176312,0.0,2.157297344537006>,0.05
    ,<0.0030464250726898725,0.0,2.2324909112466513>,0.05
    ,<0.0030462811675407123,0.0,2.307673390089592>,0.05
    ,<0.003046146003812621,0.0,2.3828447843353255>,0.05
    ,<0.0030460195789408424,0.0,2.458005097251904>,0.05
    ,<0.0030459018903495924,0.0,2.5331543321059358>,0.05
    ,<0.003045792935473406,0.0,2.6082924921625863>,0.05
    ,<0.003045692711744981,0.0,2.6834195806855785>,0.05
    ,<0.00304560121658482,0.0,2.758535600937191>,0.05
    ,<0.003045518447426281,0.0,2.8336405561782483>,0.05
    ,<0.003045444401714486,0.0,2.9087344496681666>,0.05
    ,<0.0030453790768912893,0.0,2.9838172846648896>,0.05
    ,<0.0030453224703846623,0.0,3.0588890644249362>,0.05
    ,<0.0030452745796185825,0.0,3.13394979220338>,0.05
    ,<0.003045235402030768,0.0,3.208999471253874>,0.05
    ,<0.0030452049350595083,0.0,3.2840381048286114>,0.05
    ,<0.0030451831761338644,0.0,3.359065696178369>,0.05
    ,<0.0030451701226907346,0.0,3.434082248552475>,0.05
    ,<0.0030451657721807736,0.0,3.509087765198831>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }