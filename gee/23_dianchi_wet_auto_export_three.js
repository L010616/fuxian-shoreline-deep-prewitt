// Dianchi Waihai: prediction-blind automatic search for ONE unused wet-season
// Landsat 8 / Sentinel-2 pair. Paste the entire script into the GEE Code Editor
// and click Run. It scans dates sequentially and prepares EXACTLY THREE Drive
// GeoTIFF tasks (L8, S2, QA_JRC) for the first pair passing pre-export QC.
// Then open Tasks and click Run on those three tasks. No left-side UI selection.
// If no candidate passes, it prepares ZERO tasks and prints the reason.
// This is image-quality selection only; no model output or reference is used.

var BOX = ee.Geometry.Rectangle([102.57, 24.62, 102.87, 25.06], null, false);
var CRS = 'EPSG:32648';
var FOLDER = 'Fuxian_Q2_external_candidates';
var CENTER = [102.68659, 24.82303]; // 2019 JRC Dianchi main-lake centroid
var ANCHORS = [[102.67310, 24.92293], [102.71056, 24.82385],
  [102.64620, 24.71237]];
var MAX_L8_DATES = 250;
var MAX_S2_SCENES_PER_L8 = 40;
var EXCLUDED_L8 = {
  '2020-07-28': true, // retained wet pair
  '2022-08-19': true, // retained wet pair
  '2021-10-19': true, // failed local 12-sector QC
  '2023-06-03': true, // cloud + incomplete S2 coverage
  '2024-10-27': true  // failed local 12-sector QC
};

function l8Valid(image) {
  return image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(image.select('QA_RADSAT').eq(0))
    .and(image.select('SR_B2').mask())
    .rename('l8_valid').unmask(0, false).toByte();
}
function s2Valid(image) {
  var scl = image.select('SCL');
  return scl.neq(0).and(scl.neq(1)).and(scl.neq(3))
    .and(scl.lt(8)).and(image.select('B2').mask())
    .rename('s2_valid').unmask(0, false).toByte();
}
function l8Source(day) {
  // Mosaic every WRS scene acquired on the same date; the least cloudy scene
  // has final priority in overlaps. This avoids clipping the lake to one row.
  return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(BOX)
    .filterDate(day, ee.Date(day).advance(1, 'day'))
    .sort('CLOUD_COVER', false).mosaic();
}
function s2Source(day) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(BOX)
    .filterDate(day, ee.Date(day).advance(1, 'day'));
}
var water = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first())
  .select('waterClass').gte(2).unmask(0, false);
var shore = water.focalMax({radius: 300, units: 'meters'})
  .and(water.focalMin({radius: 300, units: 'meters'}).not());
var boundary = water.and(water.focalMin({
  radius: 1, kernelType: 'square', units: 'pixels'
}).not())
  .selfMask();
var ll = ee.Image.pixelLonLat();
var dx = ll.select('longitude').subtract(CENTER[0])
  .multiply(Math.cos(CENTER[1] * Math.PI / 180));
var dy = ll.select('latitude').subtract(CENTER[1]);
var sector = dy.atan2(dx).add(Math.PI).divide(2 * Math.PI)
  .multiply(12).floor().toInt().mod(12).rename('sector');
function meanOn(image, mask) {
  return image.updateMask(mask).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: BOX, scale: 30,
    maxPixels: 1e7, tileScale: 16
  });
}
function l8Check(day) {
  var source = l8Source(day);
  var foot = source.select('SR_B2').mask().unmask(0, false)
    .rename('l8_foot').toByte();
  var valid = l8Valid(source);
  return ee.Dictionary(meanOn(foot, water))
    .combine(ee.Dictionary(meanOn(valid, shore)), true);
}
function pairCheck(l8Day, s2Day) {
  var lm = l8Valid(l8Source(l8Day));
  var sc = s2Source(s2Day);
  var sf = sc.select('B2').count().gt(0).unmask(0, false)
    .rename('s2_foot').toByte();
  var sm = sc.map(s2Valid).max().unmask(0, false)
    .rename('s2_valid').toByte();
  var joint = lm.and(sm).rename('joint_valid').toByte();
  var groups = joint.addBands(sector).updateMask(boundary)
    .reduceRegion({
      reducer: ee.Reducer.mean().group({groupField: 1, groupName: 'sector'}),
      geometry: BOX, scale: 30, maxPixels: 1e7, tileScale: 16
    });
  return ee.Dictionary(meanOn(sf, water))
    .combine(ee.Dictionary(meanOn(sm.addBands(joint), shore)), true)
    .combine(ee.Dictionary(groups), true);
}
function buildImages(l8Day, s2Day) {
  var l8SourceImage = l8Source(l8Day);
  var lm = l8Valid(l8SourceImage);
  var l8 = l8SourceImage.select(['SR_B2', 'SR_B3', 'SR_B4',
      'SR_B5', 'SR_B6', 'SR_B7'])
    .multiply(0.0000275).add(-0.2)
    .rename(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    .addBands(lm.rename('valid'))
    .toFloat().unmask(-9999).clip(BOX);
  var sc = s2Source(s2Day).map(function(image) {
    return image.addBands(s2Valid(image));
  });
  var sm = sc.select('s2_valid').max().unmask(0, false)
    .rename('s2_valid').toByte();
  var s2 = sc.qualityMosaic('s2_valid')
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL', 's2_valid'])
    .toUint16().unmask(0).clip(BOX);
  var jrc = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
    .filter(ee.Filter.eq('year', 2019)).first())
    .select('waterClass').unmask(0).toByte().rename('jrc2019_class');
  var qa = jrc.addBands(lm).addBands(sm)
    .addBands(lm.and(sm).rename('joint_valid').toByte())
    .toByte().unmask(0).clip(BOX);
  return {l8: l8, s2: s2, qa: qa};
}
function prepareExports(l8Day, s2Day, metrics) {
  var images = buildImages(l8Day, s2Day);
  var stem = 'Q2_FAST_Dianchi_Waihai_' + l8Day.replace(/-/g, '') +
    '_' + s2Day.replace(/-/g, '');
  Export.image.toDrive({image: images.l8, description: stem + '_L8',
    folder: FOLDER, region: BOX, crs: CRS,
    crsTransform: [30, 0, 0, 0, -30, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: -9999}});
  Export.image.toDrive({image: images.s2, description: stem + '_S2',
    folder: FOLDER, region: BOX, crs: CRS,
    crsTransform: [10, 0, 0, 0, -10, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: 0}});
  Export.image.toDrive({image: images.qa, description: stem + '_QA_JRC',
    folder: FOLDER, region: BOX, crs: CRS,
    crsTransform: [30, 0, 0, 0, -30, 0], maxPixels: 1e9,
    fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}});
  print('SELECTED_WET_PAIR_AND_THREE_TASKS', l8Day, s2Day, metrics);
  print('Open Tasks and start the three Q2_FAST_Dianchi_Waihai tasks. ' +
    'Download all three GeoTIFFs together; local 12-sector QC remains final.');
  Map.layers().reset();
  Map.addLayer(images.l8, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.25},
    'L8 ' + l8Day, true);
  Map.addLayer(images.s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
    'S2 ' + s2Day, false);
  Map.centerObject(BOX, 10);
}

var l8Catalog = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(BOX)
  .filterDate('2020-01-01', '2026-01-01')
  .filter(ee.Filter.calendarRange(6, 10, 'month'))
  .filter(ee.Filter.lt('CLOUD_COVER', 85));
ANCHORS.forEach(function(point) {
  l8Catalog = l8Catalog.filterBounds(ee.Geometry.Point(point));
});
l8Catalog = l8Catalog.sort('CLOUD_COVER').limit(MAX_L8_DATES);
var l8Rows = ee.FeatureCollection(l8Catalog.toList(MAX_L8_DATES).map(
  function(item) {
    var image = ee.Image(item);
    return ee.Feature(null, {
      date: ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
      scene_cloud: image.get('CLOUD_COVER')
    });
  }));

print('Scanning unused 2020-2025 wet-season dates sequentially. ' +
  'Wait for SELECTED_WET_PAIR_AND_THREE_TASKS or NO_VALID_PAIR.');
l8Rows.evaluate(function(fc, error) {
  if (error) { print('L8_CATALOG_ERROR', error); return; }
  var seen = {};
  var days = fc.features.map(function(f) { return f.properties; })
    .filter(function(row) {
      if (EXCLUDED_L8[row.date] || seen[row.date]) return false;
      seen[row.date] = true;
      return true;
    });
  print('Unused wet L8 dates in shortlist:', days.length);
  function tryL8(index) {
    if (index >= days.length) {
      print('NO_VALID_PAIR: all candidate dates failed the fixed image-quality ' +
        'screen. No export tasks were created. Do not relax the thresholds.');
      return;
    }
    var l8Day = days[index].date;
    l8Check(l8Day).evaluate(function(l8Stats, l8Error) {
      if (l8Error) {
        print('L8_CHECK_ERROR', l8Day, l8Error);
        tryL8(index + 1); return;
      }
      var l8Foot = Number(l8Stats.l8_foot || 0);
      var l8Shore = Number(l8Stats.l8_valid || 0);
      if (l8Foot < 0.98 || l8Shore < 0.50) {
        print('REJECT_L8', l8Day, 'lake footprint', l8Foot,
          'shore valid', l8Shore);
        tryL8(index + 1); return;
      }
      var day = ee.Date(l8Day);
      var s2Catalog = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(BOX)
        .filterDate(day.advance(-4, 'day'), day.advance(5, 'day'))
        .sort('CLOUDY_PIXEL_PERCENTAGE').limit(MAX_S2_SCENES_PER_L8);
      var s2Rows = ee.FeatureCollection(
        s2Catalog.toList(MAX_S2_SCENES_PER_L8).map(function(item) {
          var image = ee.Image(item);
          return ee.Feature(null, {
            date: ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
            tile_cloud: image.get('CLOUDY_PIXEL_PERCENTAGE')
          });
        }));
      s2Rows.evaluate(function(s2fc, s2Error) {
        if (s2Error) {
          print('S2_CATALOG_ERROR', l8Day, s2Error);
          tryL8(index + 1); return;
        }
        var s2Seen = {};
        var s2Days = s2fc.features.map(function(f) { return f.properties; })
          .filter(function(row) {
            if (s2Seen[row.date]) return false;
            s2Seen[row.date] = true;
            return true;
          });
        function tryS2(j) {
          if (j >= s2Days.length) { tryL8(index + 1); return; }
          var s2Day = s2Days[j].date;
          pairCheck(l8Day, s2Day).evaluate(function(stats, pairError) {
            if (pairError) {
              print('PAIR_CHECK_ERROR', l8Day, s2Day, pairError);
              tryS2(j + 1); return;
            }
            var s2Foot = Number(stats.s2_foot || 0);
            var s2Shore = Number(stats.s2_valid || 0);
            var jointShore = Number(stats.joint_valid || 0);
            if (!stats.groups || !stats.groups.length) {
              print('SECTOR_CHECK_ERROR: JRC shoreline has no grouped pixels. ' +
                'Stop; do not export or accept this screen.', stats);
              return;
            }
            var covered = stats.groups.filter(function(group) {
              return Number(group.mean || 0) >= 0.50;
            }).length;
            if (s2Foot >= 0.98 && s2Shore >= 0.50 &&
                jointShore >= 0.40 && covered >= 8) {
              prepareExports(l8Day, s2Day, {
                l8_foot: l8Foot, l8_shore_valid: l8Shore,
                s2_foot: s2Foot, s2_shore_valid: s2Shore,
                joint_shore_valid: jointShore,
                approx_sectors_of_12: covered
              });
              return;
            }
            print('REJECT_PAIR', l8Day, s2Day, 'S2 footprint', s2Foot,
              'S2/joint shore', s2Shore, jointShore,
              'approx sectors', covered);
            tryS2(j + 1);
          });
        }
        tryS2(0);
      });
    });
  }
  tryL8(0);
});
