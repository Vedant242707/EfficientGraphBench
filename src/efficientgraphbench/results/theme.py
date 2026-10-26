"""Shared screen/portrait-print styling. No effect on benchmark calculations."""


def dark_css(scope: str) -> str:
    return """
    :root{color-scheme:dark}html,body{background:#0B1020;
color:#E8EEF7;
margin:0}
    S{font:14px/1.5 'segoe UI',Arial,sans-serif!important;
background:#0B1020!important;

      color:#E8EEF7!important;
max-width:1100px;
margin:auto;
padding:28px!important}
    S h1{font-size:25px;
margin:0 0 8px;
color:#E8EEF7}
    S h2{font-size:19px;
margin:28px 0 12px;
border-bottom:1px solid #293548;
padding-bottom:8px;
color:#60A5FA}
    S h3{font-size:16px;
color:#22D3EE}S p{color:#B8C5D9}
    S h4,S label,S summary,S pre,S li,S strong,S b{color:#E8EEF7!important}
    S .criteria h4{color:#E8EEF7!important}
    S a{color:#79D5FF!important}
    S code{color:#E8EEF7!important;background:#172033!important}
    S table{width:100%;
border-collapse:collapse;
table-layout:auto;
margin:12px 0 18px!important}
    S th,S td{border:1px solid #293548!important;
padding:9px!important;
vertical-align:top!important}
    S table thead th{background:#172033!important;
color:#E8EEF7!important;
font-weight:600}
    S table tbody td{background:#111827!important;
color:#E8EEF7!important}
    S table tbody tr:nth-child(even) td{background:#141E30!important}
    S .text{text-align:left!important}S .numeric{text-align:right!important;
font-variant-numeric:tabular-nums}
    S .criteria,S .alert{background:#172033!important;
color:#E8EEF7!important;
border-color:#60A5FA!important}
    S .cards{display:flex;
gap:12px;
flex-wrap:wrap;
margin:18px 0}
    S .card{background:#111827;
border:1px solid #293548;
padding:12px 18px;
flex:1;
min-width:125px}
    S .card b{display:block;
font-size:20px;
color:#22D3EE}S .card span{color:#9CAAC0;
font-size:12px}
    S .stage-flow{display:flex;
gap:12px;
margin:16px 0}S .stage-box{flex:1;
padding:12px;
background:#172033;
border-top:2px solid #60A5FA}
    S img.chart{width:100%;
height:auto;
display:block;
margin:16px 0}
    S td.positive{color:#34D399!important}S td.negative{color:#FB7185!important}
    @media print{
      @page{size:A4 portrait;
margin:10mm;
background:#0B1020}
      *{-webkit-print-color-adjust:exact!important;
print-color-adjust:exact!important}
      body::before{content:'';
position:fixed;
inset:-20mm;
background:#0B1020;
z-index:-1}
      S{font-size:12px!important;
padding:0!important;
max-width:none}
      S h1{font-size:20px}S h2{font-size:15px;
margin-top:18px}S h3{font-size:12px}
      S th,S td{padding:6px!important;
overflow-wrap:anywhere}
      S .table-scroll{overflow:visible!important}S table{table-layout:fixed}
      S thead{display:table-header-group}S tr,S .card,S img.chart{break-inside:avoid}
      S h2,S h3,S h2+p,S h3+p{break-after:avoid}
      S .cards,S .stage-flow{break-inside:avoid}
    }
    """.replace("S", scope)
