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
    ,<0.0001642620441619877,0.0,0.10052940575180953>,0.08
    ,<0.00032851376918995013,0.0,0.20103899385465251>,0.08
    ,<0.0004927348319920917,0.0,0.30152877213273793>,0.08
    ,<0.0006569261539229586,0.0,0.4019987483932416>,0.08
    ,<0.0008210881927717608,0.0,0.5024489304389994>,0.08
    ,<0.0009852208869438868,0.0,0.6028793260685487>,0.08
    ,<0.0011493237581656447,0.0,0.7032899430760521>,0.08
    ,<0.0013133961580095153,0.0,0.8036807892511637>,0.08
    ,<0.001477437608993378,0.0,0.9040518723788433>,0.08
    ,<0.0016414481664114259,0.0,1.0044032002391374>,0.08
    ,<0.0018054287180060665,0.0,1.1047347806070174>,0.08
    ,<0.001969381144925318,0.0,1.2050466212522835>,0.08
    ,<0.0021333082895422365,0.0,1.3053387299395818>,0.08
    ,<0.002297213708927574,0.0,1.4056111144285803>,0.08
    ,<0.002461101243025637,0.0,1.505863782474173>,0.08
    ,<0.002624974450374404,0.0,1.606096741826857>,0.08
    ,<0.002788836018598837,0.0,1.706310000233062>,0.08
    ,<0.002952687254729983,0.0,1.8065035654354586>,0.08
    ,<0.003116527772662268,0.0,1.9066774451731385>,0.08
    ,<0.0032803905102388627,0.0,2.006831647161158>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0032803675688881104,0.0,2.006876935655434>,0.05
    ,<0.0032801975598940355,0.0,2.0820926879126858>,0.05
    ,<0.003280036293146514,0.0,2.1572973457599>,0.05
    ,<0.0032798837660528444,0.0,2.2324909124694687>,0.05
    ,<0.003279739976041247,0.0,2.3076733913123344>,0.05
    ,<0.0032796049205292836,0.0,2.3828447855579906>,0.05
    ,<0.0032794785969183678,0.0,2.4580050984745054>,0.05
    ,<0.003279361002626681,0.0,2.5331543333284694>,0.05
    ,<0.0032792521350945856,0.0,2.6082924933850657>,0.05
    ,<0.003279151991746765,0.0,2.6834195819080056>,0.05
    ,<0.00327906056999126,0.0,2.758535602159566>,0.05
    ,<0.003278977867250828,0.0,2.8336405574005874>,0.05
    ,<0.003278903880973971,0.0,2.9087344508904613>,0.05
    ,<0.0032788386085953426,0.0,2.983817285887147>,0.05
    ,<0.0032787820475333237,0.0,3.0588890656471635>,0.05
    ,<0.0032787341952203136,0.0,3.133949793425585>,0.05
    ,<0.00327869504909438,0.0,3.2089994724760533>,0.05
    ,<0.003278664606591284,0.0,3.28403810605078>,0.05
    ,<0.003278642865156033,0.0,3.3590656974005286>,0.05
    ,<0.0032786298222351896,0.0,3.434082249774626>,0.05
    ,<0.003278625475243092,0.0,3.5090877664209863>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
