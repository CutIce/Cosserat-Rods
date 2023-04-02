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
    ,<4.817086695436849e-05,0.0,0.10052940912058086>,0.08
    ,<9.633282027972849e-05,0.0,0.2010390006255193>,0.08
    ,<0.00014448648968113753,0.0,0.3015287823339425>,0.08
    ,<0.00019263250746294093,0.0,0.40199876203033647>,0.08
    ,<0.0002407715284278931,0.0,0.5024489474622268>,0.08
    ,<0.00028890431602898145,0.0,0.6028793464076058>,0.08
    ,<0.000337031801548438,0.0,0.7032899665422357>,0.08
    ,<0.0003851551062630256,0.0,0.803680815728566>,0.08
    ,<0.00043327551592414716,0.0,0.904051901734481>,0.08
    ,<0.00048139440462449326,0.0,1.0044032321816687>,0.08
    ,<0.000529513121568698,0.0,1.1047348146985412>,0.08
    ,<0.0005776328641329363,0.0,1.205046656902514>,0.08
    ,<0.0006257545622447571,0.0,1.3053387665135947>,0.08
    ,<0.0006738788007864062,0.0,1.4056111512676726>,0.08
    ,<0.0007220058021094026,0.0,1.5058638189713234>,0.08
    ,<0.0007701354786876017,0.0,1.6060967775149204>,0.08
    ,<0.0008182675637844819,0.0,1.706310034671175>,0.08
    ,<0.0008664017881830139,0.0,1.8065035982265083>,0.08
    ,<0.0009145380917689193,0.0,1.906677475942557>,0.08
    ,<0.0009626635048182634,0.0,2.0068316755684585>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0009626405502105945,0.0,2.0068769633221275>,0.05
    ,<0.0009624704437795738,0.0,2.0820927118695756>,0.05
    ,<0.0009623090867808702,0.0,2.1572973658696584>,0.05
    ,<0.0009621564760968604,0.0,2.2324909286133066>,0.05
    ,<0.0009620126088514235,0.0,2.307673403446265>,0.05
    ,<0.0009618774822816137,0.0,2.382844793759704>,0.05
    ,<0.00096175109316855,0.0,2.458005102910291>,0.05
    ,<0.0009616334387464288,0.0,2.533154334183266>,0.05
    ,<0.0009615245160958087,0.0,2.608292490846422>,0.05
    ,<0.0009614243217857857,0.0,2.6834195762101465>,0.05
    ,<0.0009613328526059168,0.0,2.7585355936123825>,0.05
    ,<0.0009612501056396628,0.0,2.8336405463596233>,0.05
    ,<0.0009611760779528941,0.0,2.908734437699081>,0.05
    ,<0.000961110766989549,0.0,2.9838172708390975>,0.05
    ,<0.0009610541705391263,0.0,3.05888904898719>,0.05
    ,<0.0009610062862391725,0.0,3.1339497753833614>,0.05
    ,<0.0009609671122167979,0.0,3.208999453314674>,0.05
    ,<0.000960936646514649,0.0,3.2840380860888128>,0.05
    ,<0.0009609148872115109,0.0,3.359065676970258>,0.05
    ,<0.0009609018331057872,0.0,3.43408222913961>,0.05
    ,<0.0009608974823776899,0.0,3.5090877457355134>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }