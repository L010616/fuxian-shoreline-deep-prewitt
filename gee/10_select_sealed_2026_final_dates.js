// Select three QA-only 2026 dates for the sealed final test.
// This script exports metadata only. It creates no image export and no label.
var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var water2019 = ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first().select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox, scale: 30, geometryType: 'polygon', eightConnected: true,
  maxPixels: 1e7, geometryInNativeProjection: false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000, 30);
var shore = lake.buffer(300, 30).difference(lake.buffer(-300, 30), 30);

function l8Valid(im) {
  return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(im.select('QA_RADSAT').eq(0)).and(im.select('SR_B2').mask())
    .rename('valid').unmask(0);
}
function s2Valid(im) {
  var scl = im.select('SCL');
  return scl.neq(0).and(scl.neq(1)).and(scl.neq(3)).and(scl.lt(8))
    .and(im.select('B2').mask()).rename('valid').unmask(0);
}
function clearFraction(mask, scale) {
  return ee.Number(mask.unmask(0).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: shore, scale: scale,
    maxPixels: 1e8, tileScale: 4
  }).get('valid'));
}

var start = ee.Date('2026-03-01');
var end = ee.Date('2026-10-01');
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi).filterDate(start.advance(-2, 'day'), end.advance(2, 'day'));
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi).filterDate(start, end)
  .filter(ee.Filter.eq('WRS_PATH', 129)).filter(ee.Filter.eq('WRS_ROW', 43));
var offsets = ee.List([-2, -1, 0, 1, 2]);

function bestPair(date) {
  return ee.Feature(ee.FeatureCollection(offsets.map(function(o) {
    o = ee.Number(o);
    var d = date.advance(o, 'day');
    var daily = s2.filterDate(d, d.advance(1, 'day'));
    var n = daily.size();
    var clear = ee.Number(ee.Algorithms.If(n.gt(0), clearFraction(daily.map(s2Valid).max(), 20), -1));
    return ee.Feature(null, {
      s2_date: d.format('YYYY-MM-dd'), s2_count: n,
      s2_clear_shore: clear, day_gap: o.abs(),
      pair_score: clear.subtract(o.abs().multiply(0.002))
    });
  })).sort('pair_score', false).first());
}

var list = l8.toList(l8.size());
var candidates = ee.FeatureCollection(list.map(function(obj) {
  var im = ee.Image(obj);
  var date = im.date();
  var pair = bestPair(date);
  var lc = clearFraction(l8Valid(im), 30);
  var sc = ee.Number(pair.get('s2_clear_shore'));
  var gap = ee.Number(pair.get('day_gap'));
  return ee.Feature(null, {
    l8_date: date.format('YYYY-MM-dd'),
    l8_index: im.get('system:index'),
    l8_product_id: im.get('LANDSAT_PRODUCT_ID'),
    l8_clear_shore: lc,
    s2_date: pair.get('s2_date'),
    s2_clear_shore: sc,
    s2_scene_count: pair.get('s2_count'),
    day_gap: gap,
    joint_clear_score: lc.min(sc),
    quality_score: lc.multiply(0.5).add(sc.multiply(0.5)).subtract(gap.multiply(0.002))
  });
})).filter(ee.Filter.gt('s2_scene_count', 0));

// Rank by joint shoreline coverage. Metadata only; no model output is available.
var selected = candidates.sort('joint_clear_score', false).limit(3)
  .map(function(f) { return f.set({role: 'SEALED_FINAL_TEST', imagery_opened: false}); });
print('SEALED_2026_FINAL_CANDIDATES', selected);
print('COUNT_EXPECTED_3', selected.size());
Export.table.toDrive({
  collection: selected,
  description: 'Fuxian_sealed_final_test_2026_manifest',
  folder: 'Fuxian_round4_protocol',
  fileFormat: 'CSV'
});
Map.centerObject(lake, 10);
Map.addLayer(lake, {color: '00A6D6'}, 'Lake prior');
