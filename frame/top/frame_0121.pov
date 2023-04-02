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
    ,<0.00010078747547752073,0.0,0.10052940530996299>,0.08
    ,<0.00020155594133519931,0.0,0.20103899300001585>,0.08
    ,<0.000302305800086836,0.0,0.3015287708821445>,0.08
    ,<0.0004030376609497825,0.0,0.40199874676366465>,0.08
    ,<0.000503752355289871,0.0,0.5024489284471771>,0.08
    ,<0.0006044510314461104,0.0,0.6028793237305863>,0.08
    ,<0.0007051351572715889,0.0,0.7032899404070421>,0.08
    ,<0.0008058064276758023,0.0,0.8036807862650123>,0.08
    ,<0.0009064665909864027,0.0,0.904051869088366>,0.08
    ,<0.0010071172275933882,0.0,1.0044031966564138>,0.08
    ,<0.0011077595274440316,0.0,1.1047347767440223>,0.08
    ,<0.001208394117637859,0.0,1.2050466171216598>,0.08
    ,<0.001309020984989328,0.0,1.3053387255554496>,0.08
    ,<0.0014096395233899846,0.0,1.4056111098071522>,0.08
    ,<0.0015102487147239314,0.0,1.505863777634074>,0.08
    ,<0.0016108474175512888,0.0,1.6060967367890175>,0.08
    ,<0.001711434733941426,0.0,1.7063099950201182>,0.08
    ,<0.0018120103791711012,0.0,1.8065035600707149>,0.08
    ,<0.0019125749873043793,0.0,1.9066774396792425>,0.08
    ,<0.002013104096936699,0.0,2.0068316415885885>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0020130811406054746,0.0,2.0068769300829947>,0.05
    ,<0.0020129110205482632,0.0,2.0820926823408987>,0.05
    ,<0.00201274964837342,0.0,2.1572973401887334>,0.05
    ,<0.002012597021517131,0.0,2.2324909068988603>,0.05
    ,<0.002012453137420415,0.0,2.3076733857422704>,0.05
    ,<0.0020123179934914462,0.0,2.3828447799884533>,0.05
    ,<0.002012191587134173,0.0,2.45800509290546>,0.05
    ,<0.002012073915799309,0.0,2.53315432775989>,0.05
    ,<0.0020119649769247618,0.0,2.60829248781691>,0.05
    ,<0.0020118647679143666,0.0,2.6834195763402495>,0.05
    ,<0.00201177328618885,0.0,2.75853559659216>,0.05
    ,<0.0020116905291921038,0.0,2.8336405518335055>,0.05
    ,<0.002011616494361961,0.0,2.9087344453236774>,0.05
    ,<0.0020115511791305597,0.0,2.983817280320606>,0.05
    ,<0.0020114945809366273,0.0,3.058889060080832>,0.05
    ,<0.002011446697216267,0.0,3.133949787859442>,0.05
    ,<0.0020114075253941416,0.0,3.2089994669100577>,0.05
    ,<0.0020113770628977057,0.0,3.2840381004848926>,0.05
    ,<0.0020113553071688327,0.0,3.359065691834714>,0.05
    ,<0.0020113422556532704,0.0,3.434082244208853>,0.05
    ,<0.0020113379057891574,0.0,3.5090877608552224>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }