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
    ,<6.570992749215828e-05,0.0,0.10052940452406584>,0.08
    ,<0.00013141301906066454,0.0,0.20103899143179874>,0.08
    ,<0.00019710460687742028,0.0,0.3015287685346654>,0.08
    ,<0.00026278546391327804,0.0,0.40199874364085403>,0.08
    ,<0.0003284561580415388,0.0,0.5024489245530206>,0.08
    ,<0.0003941170669010598,0.0,0.6028793190727629>,0.08
    ,<0.00045976845468439646,0.0,0.7032899349982261>,0.08
    ,<0.0005254105969419157,0.0,0.803680780125296>,0.08
    ,<0.0005910439259044874,0.0,0.9040518622412377>,0.08
    ,<0.000656669162530218,0.0,1.0044031891214569>,0.08
    ,<0.000722287402616071,0.0,1.1047347685391655>,0.08
    ,<0.0007879001315484288,0.0,1.2050466082670164>,0.08
    ,<0.0008535091533286382,0.0,1.3053387160692518>,0.08
    ,<0.0009191164421284868,0.0,1.405611099704847>,0.08
    ,<0.0009847239319825585,0.0,1.5058637669355388>,0.08
    ,<0.001050333279576401,0.0,1.6060967255188268>,0.08
    ,<0.0011159456515120526,0.0,1.7063099832045288>,0.08
    ,<0.0011815615727877938,0.0,1.8065035477352576>,0.08
    ,<0.0012471808809237902,0.0,1.9066774268542859>,0.08
    ,<0.0013128127891945006,0.0,2.0068316282977>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0013127898440119172,0.0,2.006876916669391>,0.05
    ,<0.0013126198066170684,0.0,2.0820926683337597>,0.05
    ,<0.001312458512888549,0.0,2.1572973256034986>,0.05
    ,<0.001312305960199297,0.0,2.232490891752417>,0.05
    ,<0.001312162145936876,0.0,2.3076733700529113>,0.05
    ,<0.0013120270675142864,0.0,2.382844763775924>,0.05
    ,<0.0013119007223623925,0.0,2.458005076192323>,0.05
    ,<0.0013117831079366622,0.0,2.533154310572117>,0.05
    ,<0.0013116742217392278,0.0,2.608292470181806>,0.05
    ,<0.0013115740612653444,0.0,2.6834195582837492>,0.05
    ,<0.0013114826239773118,0.0,2.75853557813884>,0.05
    ,<0.001311399907342477,0.0,2.8336405330089556>,0.05
    ,<0.0013113259088023041,0.0,2.9087344261567236>,0.05
    ,<0.0013112606257508757,0.0,2.9838172608441473>,0.05
    ,<0.001311204055586567,0.0,3.058889040331789>,0.05
    ,<0.0013111561957116207,0.0,3.1339497678781276>,0.05
    ,<0.0013111170434934637,0.0,3.208999446738419>,0.05
    ,<0.001311086596312819,0.0,3.2840380801644278>,0.05
    ,<0.001311064851573428,0.0,3.3590656714061535>,0.05
    ,<0.001311051806669222,0.0,3.434082223713838>,0.05
    ,<0.0013110474590101104,0.0,3.509087740337656>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }