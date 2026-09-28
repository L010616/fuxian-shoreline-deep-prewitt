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
