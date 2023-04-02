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
    ,<7.656713876249425e-05,0.0,0.10052940525493376>,0.08
    ,<0.0001531190362783903,0.0,0.20103899289444827>,0.08
    ,<0.00022965726532376494,0.0,0.30152877073003165>,0.08
    ,<0.000306182511325322,0.0,0.401998746568672>,0.08
    ,<0.0003826956491151927,0.0,0.5024489282127281>,0.08
    ,<0.00045919779214813615,0.0,0.60287932346034>,0.08
    ,<0.0005356902703830491,0.0,0.7032899401046704>,0.08
    ,<0.0006121745393140211,0.0,0.8036807859338371>,0.08
    ,<0.0006886520368017147,0.0,0.9040518687318676>,0.08
    ,<0.0007651240164254256,0.0,1.0044031962781534>,0.08
    ,<0.0008415913941797776,0.0,1.1047347763472053>,0.08
    ,<0.000918054645381405,0.0,1.205046616709113>,0.08
    ,<0.0009945137818286726,0.0,1.3053387251296127>,0.08
    ,<0.0010709684249981476,0.0,1.405611109369558>,0.08
    ,<0.0011474179728027205,0.0,1.5058637771852377>,0.08
    ,<0.0012238618345677787,0.0,1.6060967363292118>,0.08
    ,<0.0013002997088700952,0.0,1.7063099945489448>,0.08
    ,<0.0013767318395941472,0.0,1.806503559586905>,0.08
    ,<0.0014531592043618606,0.0,1.9066774391812846>,0.08
    ,<0.0015295655880990548,0.0,2.006831641070837>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0015295426338577517,0.0,2.006876929570077>,0.05
    ,<0.0015293725292997362,0.0,2.082092681850494>,0.05
    ,<0.0015292111718471634,0.0,2.157297339718618>,0.05
    ,<0.0015290585589342646,0.0,2.232490906446929>,0.05
    ,<0.0015289146879900444,0.0,2.3076733853064644>,0.05
    ,<0.0015287795564343527,0.0,2.382844779566985>,0.05
    ,<0.0015286531616874907,0.0,2.458005092496887>,0.05
    ,<0.0015285355011654131,0.0,2.53315432736297>,0.05
    ,<0.0015284265722833258,0.0,2.6082924874303828>,0.05
    ,<0.001528326372470485,0.0,2.683419575962848>,0.05
    ,<0.0015282348991565297,0.0,2.758535596222883>,0.05
    ,<0.0015281521497692962,0.0,2.8336405514715097>,0.05
    ,<0.001528078121732495,0.0,2.9087344449680606>,0.05
    ,<0.0015280128124746707,0.0,2.983817279970365>,0.05
    ,<0.0015279562194335532,0.0,3.0588890597349976>,0.05
    ,<0.0015279083400446258,0.0,3.1339497875172153>,0.05
    ,<0.001527869171755936,0.0,3.208999466570698>,0.05
    ,<0.0015278387120203855,0.0,3.284038100147622>,0.05
    ,<0.0015278169582627196,0.0,3.3590656914988526>,0.05
    ,<0.001527803907908583,0.0,3.434082243873833>,0.05
    ,<0.0015277995584229921,0.0,3.5090877605204907>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }