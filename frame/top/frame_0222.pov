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
    ,<0.0001851417633548455,0.0,0.10052940594375424>,0.08
    ,<0.00037026945202602273,0.0,0.20103899422950935>,0.08
    ,<0.0005553635420486176,0.0,0.30152877268249956>,0.08
    ,<0.000740425366157106,0.0,0.4019987491096305>,0.08
    ,<0.0009254561448784522,0.0,0.5024489313132752>,0.08
    ,<0.0011104566751223793,0.0,0.6028793270914713>,0.08
    ,<0.0012954271127432343,0.0,0.7032899442380601>,0.08
    ,<0.001480366913376022,0.0,0.8036807905427588>,0.08
    ,<0.0016652749542648403,0.0,0.9040518737910277>,0.08
    ,<0.001850149835285856,0.0,1.004403201763966>,0.08
    ,<0.0020349903023450496,0.0,1.1047347822379203>,0.08
    ,<0.0022197957140742803,0.0,1.205046622984285>,0.08
    ,<0.0024045664533917364,0.0,1.3053387317691807>,0.08
    ,<0.00258930418716308,0.0,1.405611116353317>,0.08
    ,<0.0027740118974066506,0.0,1.5058637844919216>,0.08
    ,<0.0029586936448885433,0.0,1.6060967439349787>,0.08
    ,<0.0031433540784940868,0.0,1.7063100024274829>,0.08
    ,<0.003327997751153503,0.0,1.8065035677099528>,0.08
    ,<0.003512628342190776,0.0,1.906677447518879>,0.08
    ,<0.0036972981704158614,0.0,2.0068316495541945>,0.08
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }

sphere_sweep {
    linear_spline 21
    ,<0.0036972752273641218,0.0,2.006876938048454>,0.05
    ,<0.003697105205723619,0.0,2.0820926903056294>,0.05
    ,<0.0036969439269299606,0.0,2.1572973481527744>,0.05
    ,<0.00369679138840515,0.0,2.2324909148622747>,0.05
    ,<0.003696647587560678,0.0,2.3076733937050755>,0.05
    ,<0.0036965125218211266,0.0,2.382844787950682>,0.05
    ,<0.0036963861886256093,0.0,2.458005100867142>,0.05
    ,<0.0036962685854092644,0.0,2.533154335721058>,0.05
    ,<0.0036961597095926846,0.0,2.6082924957776>,0.05
    ,<0.0036960595585906956,0.0,2.683419584300496>,0.05
    ,<0.0036959681298316845,0.0,2.758535604552023>,0.05
    ,<0.0036958854207584002,0.0,2.8336405597930043>,0.05
    ,<0.003695811428806851,0.0,2.9087344532828525>,0.05
    ,<0.003695746151402149,0.0,2.9838172882795053>,0.05
    ,<0.0036956895859752647,0.0,3.058889068039496>,0.05
    ,<0.003695641729967225,0.0,3.133949795817899>,0.05
    ,<0.0036956025808165876,0.0,3.208999474868358>,0.05
    ,<0.003695572135957941,0.0,3.2840381084430668>,0.05
    ,<0.003695550392834361,0.0,3.359065699792811>,0.05
    ,<0.003695537348890307,0.0,3.4340822521669074>,0.05
    ,<0.003695533001552899,0.0,3.509087768813263>,0.05
    texture{
        pigment{ color rgb<0.45,0.5,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }