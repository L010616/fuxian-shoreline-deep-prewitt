// Fuxian Lake: select previously unused Landsat 8 / Sentinel-2 test pairs.
// Selection uses image quality and temporal distance only, never model accuracy.
var seed = ee.Geometry.Point([102.8889, 24.5022]);
var water2019 = ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first().select('waterClass').gte(2).selfMask();
var components = water2019.reduceToVectors({
  geometry: seed.buffer(80000), scale: 30, geometryType: 'polygon',
  eightConnected: true, maxPixels: 1e9
});
var lake = ee.Feature(components.filterBounds(seed).first()).geometry().simplify(30);
var roi = lake.buffer(1000);
var shore = lake.buffer(300).difference(lake.buffer(-300));

var used = ee.List([
  '2020-01-02','2020-05-09','2021-02-05','2021-05-12','2022-03-12','2022-08-19',
  '2023-01-26','2023-03-31','2024-02-14','2025-05-07','2025-12-01'
]);

function l8Clear(image) {
  var qa = image.select('QA_PIXEL');
  return qa.bitwiseAnd(1 << 0).eq(0)  // fill
    .and(qa.bitwiseAnd(1 << 1).eq(0)) // dilated cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0)) // cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0)) // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0)) // cloud shadow
    .and(qa.bitwiseAnd(1 << 5).eq(0)); // snow
}

function s2Clear(image) {
  var scl = image.select('SCL');
  return scl.neq(1).and(scl.neq(3)).and(scl.neq(8)).and(scl.neq(9))
    .and(scl.neq(10)).and(scl.neq(11));
}

function fraction(mask, geometry, scale) {
  return ee.Number(mask.rename('clear').reduceRegion({
    reducer: ee.Reducer.mean(), geometry: geometry, scale: scale,
    maxPixels: 1e8, tileScale: 4
  }).get('clear'));
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi).filterDate('2023-12-01', '2026-01-01');
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi).filterDate('2024-01-01', '2026-01-01')
  .filter(ee.Filter.lt('CLOUD_COVER', 70));

var candidates = ee.FeatureCollection(l8.map(function(image) {
  var date = image.date();
  var nearby = s2.filterDate(date.advance(-2, 'day'), date.advance(3, 'day'))
    .map(function(x) {
      return x.set('day_gap', x.date().difference(date, 'day').abs());
    }).sort('day_gap');
  var hasPair = nearby.size().gt(0);
  var best = ee.Image(ee.Algorithms.If(hasPair, nearby.first(), s2.first()));
  var l8Fraction = fraction(l8Clear(image), shore, 30);
  var s2Fraction = ee.Number(ee.Algorithms.If(hasPair, fraction(s2Clear(best), shore, 20), -1));
  var gap = ee.Number(ee.Algorithms.If(hasPair, best.get('day_gap'), 99));
  var dateString = date.format('YYYY-MM-dd');
  var unused = ee.Number(ee.Algorithms.If(used.contains(dateString), 0, 1));
  var score = l8Fraction.multiply(0.55).add(s2Fraction.max(0).multiply(0.40))
    .subtract(gap.multiply(0.025));
  return ee.Feature(null, {
    l8_date: dateString,
    l8_id: image.get('LANDSAT_PRODUCT_ID'),
    l8_scene_cloud_pct: image.get('CLOUD_COVER'),
    l8_clear_shore: l8Fraction,
    s2_date: ee.Algorithms.If(hasPair, best.date().format('YYYY-MM-dd'), 'NONE'),
    s2_id: ee.Algorithms.If(hasPair, best.get('PRODUCT_ID'), 'NONE'),
    s2_clear_shore: s2Fraction,
    day_gap: gap,
    unused: unused,
    quality_score: score
  });
})).filter(ee.Filter.eq('unused', 1))
  .filter(ee.Filter.gte('l8_clear_shore', 0.90))
  .filter(ee.Filter.gte('s2_clear_shore', 0.90))
  .filter(ee.Filter.lte('day_gap', 2))
  .sort('quality_score', false);

print('UNSEEN_TEST_CANDIDATES', candidates);
print('CANDIDATE_COUNT', candidates.size());
Map.centerObject(lake, 10);
Map.addLayer(lake, {color: '00A6D6'}, 'Fuxian Lake 2019 water extent');
Map.addLayer(shore, {color: 'FF8C00'}, 'Evaluation shore band', false);
