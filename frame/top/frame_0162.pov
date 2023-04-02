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
    ,<0.0001350304291437205,0.0,0.10052940552166725>,0.08
    ,<0.00027004716096223354,0.0,0.20103899340963058>,0.08
    ,<0.00040503982401321877,0.0,0.3015287714810985>,0.08
    ,<0.0005400096067304425,0.0,0.40199874754313386>,0.08
    ,<0.000674957642836968,0.0,0.5024489293982131>,0.08
    ,<0.0008098847796317266,0.0,0.6028793248443292>,0.08
    ,<0.0009447914063922669,0.0,0.7032899416750903>,0.08
    ,<0.0010796773940549911,0.0,0.8036807876797496>,0.08
    ,<0.0012145421646019554,0.0,0.9040518706431697>,0.08
    ,<0.0013493848946776227,0.0,1.0044031983457344>,0.08
    ,<0.0014842048166747968,0.0,1.1047347785632164>,0.08
    ,<0.0016190015611531152,0.0,1.205046619066597>,0.08
    ,<0.0017537754703756089,0.0,1.305338727621933>,0.08
    ,<0.0018885278106539275,0.0,1.4056111119902481>,0.08
    ,<0.002023260824537704,0.0,1.5058637799275358>,0.08
    ,<0.0021579775885234966,0.0,1.6060967391847991>,0.08
    ,<0.0022926816808623075,0.0,1.7063099975082354>,0.08
    ,<0.0024273766976377837,0.0,1.8065035626394594>,0.08
    ,<0.002562065685194776,0.0,1.9066774423157844>,0.08
    ,<0.002696786308147533,0.0,2.006831644253326>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.002696763363367231,0.0,2.0068769327476312>,0.05
    ,<0.0026965933289195515,0.0,2.0820926850049988>,0.05
    ,<0.002696432037964896,0.0,2.1572973428523246>,0.05
    ,<0.0026962794879233434,0.0,2.232490909561998>,0.05
    ,<0.0026961356762182412,0.0,2.307673388404954>,0.05
    ,<0.0026960006002719654,0.0,2.3828447826507086>,0.05
    ,<0.0026958742575116787,0.0,2.4580050955673056>,0.05
    ,<0.002695756645375535,0.0,2.5331543304213535>,0.05
    ,<0.00269564776129914,0.0,2.6082924904780254>,0.05
    ,<0.0026955476027019708,0.0,2.683419579001036>,0.05
    ,<0.0026954561670043315,0.0,2.7585355992526592>,0.05
    ,<0.002695373451648084,0.0,2.833640554493732>,0.05
    ,<0.0026952994540805656,0.0,2.90873444798366>,0.05
    ,<0.0026952341717316653,0.0,2.983817282980394>,0.05
    ,<0.0026951776020229584,0.0,3.0588890627404495>,0.05
    ,<0.0026951297423871254,0.0,3.1339497905189053>,0.05
    ,<0.0026950905902670846,0.0,3.2089994695694033>,0.05
    ,<0.0026950601431017366,0.0,3.284038103144148>,0.05
    ,<0.002695038398326619,0.0,3.3590656944939137>,0.05
    ,<0.0026950253533824537,0.0,3.4340822468680234>,0.05
    ,<0.0026950210057083645,0.0,3.5090877635143802>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }