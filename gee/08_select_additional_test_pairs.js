// Select additional independent Landsat 8 / Sentinel-2 test pairs for Fuxian Lake.
// Read-only screening: no export tasks are created by this script.
var seed = ee.Geometry.Point([102.8889, 24.5022]);
var searchBox = ee.Geometry.Rectangle([102.78, 24.28, 103.00, 24.72], null, false);
var water2019 = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first()).select('waterClass').gte(2);
var components = water2019.selfMask().toInt().reduceToVectors({
  geometry: searchBox, scale: 30, geometryType: 'polygon', eightConnected: true,
  maxPixels: 1e7, geometryInNativeProjection: false
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var shore = lake.buffer(300, 30).difference(lake.buffer(-300, 30), 30);

function l8Valid(im) {
  return im.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(im.select('QA_RADSAT').eq(0)).and(im.select('SR_B2').mask()).unmask(0);
}
function s2Valid(im) {
  var sc = im.select('SCL');
  return sc.neq(0).and(sc.neq(1)).and(sc.neq(3)).and(sc.lt(8))
    .and(im.select('B2').mask()).unmask(0);
}
function meanOnShore(mask) {
  return ee.Number(mask.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: shore, scale: 30,
    bestEffort: false, maxPixels: 1e7, tileScale: 2
  }).values().get(0));
}
var offsets = ee.List.sequence(-2, 2);
var excluded = ee.List([
  '2024-02-14', '2024-05-04', '2025-05-07', '2025-12-01'
]);
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(seed)
  .filterDate('2024-06-01', '2026-09-01')
  .filter(ee.Filter.eq('WRS_PATH', 129))
  .filter(ee.Filter.eq('WRS_ROW', 43));

var scored = ee.FeatureCollection(l8.map(function(im) {
  var d = ee.Date(im.get('system:time_start'));
  var candidates = ee.FeatureCollection(offsets.map(function(offset) {
    offset = ee.Number(offset);
    var day = d.advance(offset, 'day');
    var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(searchBox).filterDate(day, day.advance(1, 'day'));
    var count = col.size();
    var valid = ee.Image(ee.Algorithms.If(count.gt(0), col.map(s2Valid).max(), ee.Image(0)));
    return ee.Feature(null, {
      offset: offset,
      s2_date: day.format('YYYY-MM-dd'),
      s2_count: count,
      s2_valid: meanOnShore(valid)
    });
  })).sort('s2_valid', false);
  var best = ee.Feature(candidates.first());
  var lv = l8Valid(im);
  var bestDay = ee.Date(best.get('s2_date'));
  var bestS2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(searchBox).filterDate(bestDay, bestDay.advance(1, 'day'));
  var sv = ee.Image(bestS2.map(s2Valid).max());
  var dateText = d.format('YYYY-MM-dd');
  return ee.Feature(null, {
    l8_date: dateText,
    l8_id: im.get('LANDSAT_PRODUCT_ID'),
    l8_system_index: im.get('system:index'),
    l8_valid: meanOnShore(lv),
    s2_date: best.get('s2_date'),
    gap_days: best.get('offset'),
    s2_count: best.get('s2_count'),
    s2_valid: best.get('s2_valid'),
    joint_valid: meanOnShore(lv.and(sv)),
    excluded_existing_test: excluded.contains(dateText)
  });
})).filter(ee.Filter.eq('excluded_existing_test', false))
  .filter(ee.Filter.gte('joint_valid', 0.97))
  .sort('joint_valid', false);

print('ADDITIONAL_TEST_PAIR_CANDIDATES_TOP20', scored.limit(20));
print('CANDIDATE_COUNT_GE_97_PERCENT', scored.size());
Map.centerObject(lake, 11);
Map.addLayer(ee.Image().byte().paint(lake, 1, 2), {palette:['ffff00']}, '2019 prior');
