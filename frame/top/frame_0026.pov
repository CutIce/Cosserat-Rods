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
    ,<2.1436228524049706e-05,0.0,0.10052990845880704>,0.08
    ,<4.2869083881946655e-05,0.0,0.20103997463228024>,0.08
    ,<6.429917768729956e-05,0.0,0.3015301738486217>,0.08
    ,<8.572720671550488e-05,0.0,0.4020005193048565>,0.08
    ,<0.00010715387758786415,0.0,0.50245101768679>,0.08
    ,<0.00012857997291458093,0.0,0.6028816717402404>,0.08
    ,<0.00015000622810715965,0.0,0.703292500903661>,0.08
    ,<0.0001714335911345271,0.0,0.8036835267593453>,0.08
    ,<0.0001928630549948833,0.0,0.9040547450300996>,0.08
    ,<0.00021429574018101986,0.0,1.0044061527751358>,0.08
    ,<0.00023573274370791163,0.0,1.1047377622531058>,0.08
    ,<0.00025717513951003135,0.0,1.205049566372238>,0.08
    ,<0.00027862393924101865,0.0,1.3053415825924817>,0.08
    ,<0.00030007994834315647,0.0,1.4056138114550256>,0.08
    ,<0.000321543798880606,0.0,1.5058662661432527>,0.08
    ,<0.00034301605906557705,0.0,1.6060989935179846>,0.08
    ,<0.00036449696234211286,0.0,1.7063120233935725>,0.08
    ,<0.0003859867848016655,0.0,1.8065053835573988>,0.08
    ,<0.00040748577225805234,0.0,1.906679083046003>,0.08
    ,<0.00042898910938363866,0.0,2.0068331395337253>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0004289652268796264,0.0,2.006878392524786>,0.05
    ,<0.00042878814479988,0.0,2.082093989408473>,0.05
    ,<0.00042861996101511697,0.0,2.1572985262115476>,0.05
    ,<0.0004284607221612138,0.0,2.232491984870685>,0.05
    ,<0.00042831048407001763,0.0,2.3076743463580036>,0.05
    ,<0.00042816922214087476,0.0,2.3828456053254254>,0.05
    ,<0.00042803701095089127,0.0,2.458005768633321>,0.05
    ,<0.00042791390594047756,0.0,2.533154843153147>,0.05
    ,<0.00042779982752334404,0.0,2.6082928254095146>,0.05
    ,<0.00042769481206782,0.0,2.6834197033299048>,0.05
    ,<0.00042759889082842823,0.0,2.7585354706382437>,0.05
    ,<0.0004275120007526349,0.0,2.833640139010961>,0.05
    ,<0.00042743413254530164,0.0,2.9087337319338764>,0.05
    ,<0.0004273653446796664,0.0,2.983816267620276>,0.05
    ,<0.0004273056652385262,0.0,3.0588877543494473>,0.05
    ,<0.00042725506893112115,0.0,3.1339482017323324>,0.05
    ,<0.00042721357745302617,0.0,3.208997626712879>,0.05
    ,<0.00042718128182448443,0.0,3.2840360454911055>,0.05
    ,<0.00042715821532828745,0.0,3.3590634687916476>,0.05
    ,<0.0004271443385119254,0.0,3.434079909654543>,0.05
    ,<0.0004271396921937291,0.0,3.509085386432677>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }