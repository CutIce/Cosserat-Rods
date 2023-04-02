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
    ,<0.0002369236660796333,0.0,0.10052940651881195>,0.08
    ,<0.00047383890166942573,0.0,0.20103899534065336>,0.08
    ,<0.000710710139882647,0.0,0.30152877430802405>,0.08
    ,<0.0009475386869141662,0.0,0.40199875122777146>,0.08
    ,<0.0011843253942946817,0.0,0.5024489339025262>,0.08
    ,<0.001421070381607286,0.0,0.6028793301309082>,0.08
    ,<0.0016577729528931087,0.0,0.7032899477076257>,0.08
    ,<0.0018944317499823404,0.0,0.8036807944233401>,0.08
    ,<0.002131045119376262,0.0,0.9040518780643396>,0.08
    ,<0.0023676116264148652,0.0,1.004403206412149>,0.08
    ,<0.0026041306130118657,0.0,1.1047347872430426>,0.08
    ,<0.0028406026764203917,0.0,1.2050466283276462>,0.08
    ,<0.0030770299522532266,0.0,1.305338737430698>,0.08
    ,<0.0033134161133077443,0.0,1.4056111223110548>,0.08
    ,<0.00354976604368714,0.0,1.505863790721999>,0.08
    ,<0.003786085212097191,0.0,1.6060967504117172>,0.08
    ,<0.004022378833376457,0.0,1.7063100091240682>,0.08
    ,<0.004258650943457603,0.0,1.8065035745992901>,0.08
    ,<0.004494903567608613,0.0,1.9066774545747809>,0.08
    ,<0.004731201878105273,0.0,2.0068316567301374>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.00473117893971652,0.0,2.0068769452243567>,0.05
    ,<0.004731008952653053,0.0,2.082092697481325>,0.05
    ,<0.00473084770669936,0.0,2.1572973553282684>,0.05
    ,<0.00473069519928422,0.0,2.232490922037579>,0.05
    ,<0.004730551427815668,0.0,2.307673400880199>,0.05
    ,<0.004730416389693719,0.0,2.382844795125635>,0.05
    ,<0.004730290082332534,0.0,2.458005108041935>,0.05
    ,<0.004730172503163286,0.0,2.533154342895712>,0.05
    ,<0.0047300636496240365,0.0,2.608292502952123>,0.05
    ,<0.0047299635191515015,0.0,2.683419591474892>,0.05
    ,<0.00472987210916914,0.0,2.758535611726294>,0.05
    ,<0.004729789417085702,0.0,2.8336405669671727>,0.05
    ,<0.004729715440322139,0.0,2.9087344604569294>,0.05
    ,<0.004729650176325077,0.0,2.983817295453501>,0.05
    ,<0.004729593622537918,0.0,3.0588890752134197>,0.05
    ,<0.004729545776383692,0.0,3.133949802991771>,0.05
    ,<0.004729506635284912,0.0,3.2089994820421825>,0.05
    ,<0.004729476196681175,0.0,3.284038115616856>,0.05
    ,<0.004729454458021366,0.0,3.359065706966562>,0.05
    ,<0.004729441416748251,0.0,3.434082259340637>,0.05
    ,<0.004729437070297713,0.0,3.509087775986985>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }