// Fuxian Lake pilot: inventory and pre-2020 screening geometry.
// The 2019 JRC outline is a screening prior, NEVER a reference label.
// No full-period maximum extent is used. No export is started automatically.
var cfg = {start: '2020-01-01', end: '2026-01-01', year: 2020};
var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var water2019 = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first()).select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox, scale: 30, geometryType: 'polygon',
  eightConnected: true, maxPixels: 1e7, geometryInNativeProjection: false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000, 30);
var shore = lake.buffer(300, 30).difference(lake.buffer(-300, 30), 30);
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(seed).filterDate(cfg.start, cfg.end);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi).filterDate(cfg.start, cfg.end);
function l8valid(im) {
  return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(im.select('QA_RADSAT').eq(0))
    .and(im.select('SR_B2').mask()).rename('valid').unmask(0);
}

var panel=ui.Panel({style:{width:"370px",padding:"8px"}}); ui.root.insert(0,panel); panel.add(ui.Label("Fuxian pilot: paired image screening")); Map.setCenter(102.8889,24.5022,10);
var allQuality=ee.FeatureCollection([{"date":"2020-01-02","id":"LC08_129043_20200102","month":1,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":19.9,"shore_valid":0.9723813343723652,"year":2020},{"date":"2020-01-18","id":"LC08_129043_20200118","month":1,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":0.03,"shore_valid":0.999983213819016,"year":2020},{"date":"2020-02-03","id":"LC08_129043_20200203","month":2,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":41.17,"shore_valid":0.8459850121127109,"year":2020},{"date":"2020-02-19","id":"LC08_129043_20200219","month":2,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":16.46,"shore_valid":0.8840060411819666,"year":2020},{"date":"2020-03-06","id":"LC08_129043_20200306","month":3,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":3.17,"shore_valid":0.8730885723827029,"year":2020},{"date":"2020-05-09","id":"LC08_129043_20200509","month":5,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":3.67,"shore_valid":0.9994321662856203,"year":2020},{"date":"2020-07-28","id":"LC08_129043_20200728","month":7,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":18.41,"shore_valid":0.8851265023055989,"year":2020},{"date":"2021-01-04","id":"LC08_129043_20210104","month":1,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":20.89,"shore_valid":0.9937568572372171,"year":2021},{"date":"2021-02-05","id":"LC08_129043_20210205","month":2,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":0.61,"shore_valid":0.9999328552760645,"year":2021},{"date":"2021-03-25","id":"LC08_129043_20210325","month":3,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":0.58,"shore_valid":0.9889725323441731,"year":2021},{"date":"2021-04-10","id":"LC08_129043_20210410","month":4,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":14.02,"shore_valid":0.9342497798212592,"year":2021},{"date":"2021-05-12","id":"LC08_129043_20210512","month":5,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":30.42,"shore_valid":0.994737828488282,"year":2021},{"date":"2021-10-19","id":"LC08_129043_20211019","month":10,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":33.45,"shore_valid":0.8634209170428915,"year":2021},{"date":"2021-11-20","id":"LC08_129043_20211120","month":11,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":0.04,"shore_valid":0.999983213819016,"year":2021},{"date":"2021-12-22","id":"LC08_129043_20211222","month":12,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":33.74,"shore_valid":0.8114767053558515,"year":2021},{"date":"2022-01-07","id":"LC08_129043_20220107","month":1,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":16.77,"shore_valid":0.8324554818960377,"year":2022},{"date":"2022-02-08","id":"LC08_129043_20220208","month":2,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":2.95,"shore_valid":1,"year":2022},{"date":"2022-03-12","id":"LC08_129043_20220312","month":3,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":0.04,"shore_valid":0.9999328552760645,"year":2022},{"date":"2022-04-13","id":"LC08_129043_20220413","month":4,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":1.14,"shore_valid":0.9998153520091774,"year":2022},{"date":"2022-04-29","id":"LC08_129043_20220429","month":4,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":27.61,"shore_valid":0.9911401220203954,"year":2022},{"date":"2022-08-19","id":"LC08_129043_20220819","month":8,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":20.11,"shore_valid":0.973686246002338,"year":2022},{"date":"2022-11-07","id":"LC08_129043_20221107","month":11,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":22.64,"shore_valid":0.8269972741216928,"year":2022},{"date":"2022-11-23","id":"LC08_129043_20221123","month":11,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":9.09,"shore_valid":0.9561716105919357,"year":2022},{"date":"2022-12-09","id":"LC08_129043_20221209","month":12,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":10.14,"shore_valid":0.999983213819016,"year":2022},{"date":"2023-01-10","id":"LC08_129043_20230110","month":1,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":15.56,"shore_valid":0.8658512269085514,"year":2023},{"date":"2023-01-26","id":"LC08_129043_20230126","month":1,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":0.04,"shore_valid":0.9999328552760646,"year":2023},{"date":"2023-02-11","id":"LC08_129043_20230211","month":2,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":0.11,"shore_valid":0.9983549542635813,"year":2023},{"date":"2023-03-31","id":"LC08_129043_20230331","month":3,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":5.08,"shore_valid":0.8982366413103176,"year":2023},{"date":"2023-05-18","id":"LC08_129043_20230518","month":5,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":8.29,"shore_valid":0.8959092208598652,"year":2023},{"date":"2023-11-10","id":"LC08_129043_20231110","month":11,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":15.81,"shore_valid":0.8728136739835323,"year":2023},{"date":"2024-01-13","id":"LC08_129043_20240113","month":1,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":11.35,"shore_valid":0.9294056171301196,"year":2024},{"date":"2024-01-29","id":"LC08_129043_20240129","month":1,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":37.84,"shore_valid":0.9811265396959938,"year":2024},{"date":"2024-02-14","id":"LC08_129043_20240214","month":2,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":0.88,"shore_valid":0.9982878095396458,"year":2024},{"date":"2024-04-02","id":"LC08_129043_20240402","month":4,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":4.2,"shore_valid":0.8878348703221441,"year":2024},{"date":"2024-05-04","id":"LC08_129043_20240504","month":5,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":1.29,"shore_valid":0.9820598513586963,"year":2024},{"date":"2024-08-24","id":"LC08_129043_20240824","month":8,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":45.48,"shore_valid":0.9555397260771747,"year":2024},{"date":"2024-10-27","id":"LC08_129043_20241027","month":10,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":28.89,"shore_valid":0.8639821679411998,"year":2024},{"date":"2024-12-30","id":"LC08_129043_20241230","month":12,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":12.48,"shore_valid":0.9336984689752205,"year":2024},{"date":"2025-01-31","id":"LC08_129043_20250131","month":1,"path":129,"row":43,"s2_nearby_granules":4,"scene_cloud":8.47,"shore_valid":0.9002659260208649,"year":2025},{"date":"2025-03-04","id":"LC08_129043_20250304","month":3,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":0.99,"shore_valid":0.9954303407390671,"year":2025},{"date":"2025-05-07","id":"LC08_129043_20250507","month":5,"path":129,"row":43,"s2_nearby_granules":8,"scene_cloud":5.01,"shore_valid":0.9995264980399335,"year":2025},{"date":"2025-12-01","id":"LC08_129043_20251201","month":12,"path":129,"row":43,"s2_nearby_granules":7,"scene_cloud":1.04,"shore_valid":0.9978345826530814,"year":2025}].map(function(p){return ee.Feature(null,p);}));
// Append after 01_inventory.js. S2 QA is screening, not independent truth.
var pairStatus=ui.Label('Computing near-date Sentinel-2 quality...'); panel.add(pairStatus);
var pairOutput=ui.Textbox({placeholder:'PAIR_QUALITY_JSON',style:{stretch:'horizontal'}});
panel.add(pairOutput);
function s2valid(im){
  var sc=im.select('SCL');
  // Keep dark pixels (2) and unclassified (7) for later visual inspection.
  var good=sc.neq(0).and(sc.neq(1)).and(sc.neq(3)).and(sc.lt(8));
  return good.and(im.select('B2').mask()).rename('s2_valid').unmask(0);
}
var eligible=allQuality.filter(ee.Filter.gte('shore_valid',0.8));
var pairs=ee.FeatureCollection(eligible.toList(eligible.size()).map(function(obj){
  var rec=ee.Feature(obj);
  var im=ee.Image(l8.filter(ee.Filter.eq('system:index',rec.get('id'))).first());
  var day=ee.Date.parse('YYYY-MM-dd',rec.getString('date'));
  var near=s2.filterDate(day.advance(-3,'day'),day.advance(4,'day')).map(function(s){
    return s.set('day',s.date().format('YYYY-MM-dd'));
  });
  var days=ee.List(near.aggregate_array('day')).distinct().sort();
  return ee.FeatureCollection(days.map(function(d){
    var ds=ee.String(d), images=near.filter(ee.Filter.eq('day',ds));
    var sv=images.map(s2valid).max().unmask(0);
    var jv=sv.and(l8valid(im)).rename('joint_valid');
    var vals=sv.addBands(jv).reduceRegion({reducer:ee.Reducer.mean(),geometry:shore,
      crs:'EPSG:32648',scale:30,maxPixels:1e7,tileScale:2});
    var mins=ee.Number(images.aggregate_min('system:time_start')).subtract(im.date().millis()).divide(3600000);
    return ee.Feature(null,{l8_id:rec.get('id'),l8_date:rec.get('date'),year:rec.get('year'),
      l8_shore_valid:rec.get('shore_valid'),s2_date:ds,
      calendar_gap_days:ee.Date.parse('YYYY-MM-dd',ds).difference(day,'day').abs(),
      signed_time_gap_hours:mins,s2_granule_ids:images.aggregate_array('system:index'),
      s2_shore_valid:vals.get('s2_valid'),joint_shore_valid:vals.get('joint_valid')});
  }));
})).flatten();
pairs.evaluate(function(result,error){
  if(error){pairStatus.setValue('ERROR: '+error);return;}
  var rows=result.features.map(function(f){return f.properties;});
  pairOutput.setValue(JSON.stringify(rows));print('PAIR_QUALITY',rows);
  pairStatus.setValue('Pair screening complete: '+rows.length+' date pairs. Review before labeling.');
});
