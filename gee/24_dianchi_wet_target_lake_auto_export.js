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
var TARGET_OUTLINE = [[102.671038,25.024574],[102.669275,25.023463],[102.665101,25.02421],[102.66278,25.021195],[102.659845,25.019253],[102.657535,25.015697],[102.653048,25.017251],[102.652511,25.014264],[102.650457,25.012877],[102.648659,25.013661],[102.64661,25.012004],[102.643603,25.013852],[102.640919,25.014351],[102.640299,25.015695],[102.638225,25.015391],[102.638251,25.014038],[102.640049,25.013254],[102.637406,25.011587],[102.634727,25.011815],[102.634707,25.012898],[102.632919,25.013141],[102.632638,25.012324],[102.634123,25.012347],[102.634441,25.011269],[102.633544,25.011526],[102.632382,25.010153],[102.631486,25.01041],[102.631762,25.011497],[102.630589,25.010666],[102.631215,25.009052],[102.633596,25.008819],[102.630933,25.008235],[102.630308,25.00985],[102.62913,25.009289],[102.629458,25.00767],[102.630938,25.007964],[102.630964,25.006611],[102.629489,25.006046],[102.630427,25.003623],[102.632804,25.003661],[102.629864,25.00199],[102.629896,25.000366],[102.632293,24.999321],[102.632042,24.99688],[102.630865,24.996319],[102.633543,24.996091],[102.630578,24.995773],[102.632459,24.990658],[102.631286,24.989827],[102.630093,24.990079],[102.629775,24.991157],[102.628587,24.991138],[102.628608,24.990055],[102.62829,24.991133],[102.62681,24.990839],[102.626836,24.989485],[102.628618,24.989514],[102.628639,24.988431],[102.625101,24.987021],[102.627206,24.9857],[102.628764,24.981934],[102.63176,24.980628],[102.631478,24.979811],[102.633927,24.976059],[102.636308,24.975826],[102.636943,24.97367],[102.637839,24.973414],[102.636703,24.970688],[102.637901,24.970165],[102.638229,24.968546],[102.636151,24.968513],[102.640344,24.966684],[102.639479,24.965316],[102.637994,24.965293],[102.638051,24.962315],[102.641349,24.960742],[102.641041,24.961279],[102.646069,24.962442],[102.647502,24.965172],[102.648992,24.964925],[102.648142,24.962745],[102.64312,24.961312],[102.641088,24.958843],[102.640489,24.959104],[102.642686,24.952911],[102.642314,24.941261],[102.6441,24.941019],[102.644725,24.939404],[102.646507,24.939432],[102.646251,24.937262],[102.647147,24.937005],[102.645703,24.934816],[102.643333,24.934508],[102.641518,24.920669],[102.64245,24.918517],[102.644241,24.918004],[102.643965,24.916916],[102.645454,24.916669],[102.645521,24.91315],[102.6485,24.912655],[102.650306,24.91133],[102.650629,24.909981],[102.655685,24.909519],[102.661074,24.907167],[102.661458,24.902569],[102.656739,24.90087],[102.656171,24.899508],[102.654907,24.887844],[102.65524,24.885953],[102.657343,24.884633],[102.657675,24.882742],[102.659169,24.882224],[102.660095,24.880343],[102.661876,24.880371],[102.663088,24.879036],[102.662515,24.877944],[102.658397,24.875984],[102.66053,24.873039],[102.657588,24.871639],[102.657639,24.868931],[102.662192,24.863587],[102.664576,24.863083],[102.664647,24.859293],[102.66677,24.856889],[102.665895,24.856063],[102.665669,24.852268],[102.666334,24.848487],[102.665178,24.846845],[102.66642,24.843885],[102.664942,24.843591],[102.663796,24.841407],[102.664747,24.838173],[102.663284,24.837066],[102.66334,24.834089],[102.662465,24.833263],[102.663089,24.831648],[102.655795,24.825034],[102.65556,24.821781],[102.654685,24.820955],[102.656582,24.814757],[102.658372,24.814243],[102.658106,24.812614],[102.656352,24.811233],[102.657645,24.805566],[102.656172,24.805002],[102.655896,24.803914],[102.656883,24.798785],[102.657783,24.798257],[102.656944,24.795536],[102.658201,24.791765],[102.654668,24.790355],[102.653523,24.788171],[102.650537,24.789207],[102.649668,24.788111],[102.64729,24.788344],[102.645235,24.787228],[102.643195,24.785301],[102.644166,24.780983],[102.642157,24.777432],[102.639187,24.777655],[102.637429,24.776545],[102.635649,24.776516],[102.634141,24.777847],[102.628528,24.776675],[102.625558,24.776898],[102.623237,24.774154],[102.621152,24.774662],[102.619326,24.77707],[102.616949,24.777303],[102.616382,24.77594],[102.614306,24.775907],[102.611815,24.782095],[102.610951,24.780728],[102.60976,24.780979],[102.605848,24.783896],[102.60408,24.783326],[102.604059,24.784408],[102.599939,24.782718],[102.605874,24.782542],[102.607992,24.78041],[102.608361,24.776624],[102.606311,24.775238],[102.606925,24.774164],[102.604875,24.772778],[102.606056,24.773067],[102.607273,24.771462],[102.609354,24.771225],[102.610587,24.768807],[102.611783,24.768285],[102.611502,24.767468],[102.609427,24.767435],[102.607081,24.766043],[102.606004,24.76034],[102.608095,24.759561],[102.60785,24.756849],[102.601073,24.754574],[102.600215,24.752936],[102.602035,24.750799],[102.60177,24.74917],[102.60268,24.748101],[102.608078,24.744938],[102.606638,24.742749],[102.603127,24.740255],[102.602581,24.73781],[102.60347,24.737824],[102.602893,24.737002],[102.603481,24.737282],[102.604697,24.735677],[102.603501,24.7362],[102.602929,24.735107],[102.605311,24.734604],[102.605592,24.735421],[102.605628,24.733526],[102.606523,24.733269],[102.605945,24.732448],[102.608358,24.73032],[102.608077,24.729503],[102.617374,24.723964],[102.616193,24.723675],[102.615054,24.72122],[102.612964,24.721999],[102.613593,24.720113],[102.61181,24.720356],[102.610088,24.71735],[102.609194,24.717606],[102.608949,24.714894],[102.609844,24.714638],[102.609885,24.712472],[102.607025,24.707011],[102.606141,24.706726],[102.605881,24.704826],[102.603261,24.702347],[102.600308,24.701758],[102.5988,24.703088],[102.598816,24.702276],[102.597043,24.701977],[102.596201,24.699526],[102.598291,24.698747],[102.59801,24.697931],[102.596825,24.697912],[102.598618,24.697128],[102.598078,24.694411],[102.599569,24.693894],[102.599938,24.690109],[102.600838,24.689581],[102.602298,24.690688],[102.603806,24.689358],[102.603239,24.687995],[102.604726,24.687748],[102.604455,24.68639],[102.606249,24.685606],[102.605973,24.684518],[102.607444,24.685083],[102.607163,24.684267],[102.608665,24.683207],[102.610126,24.684314],[102.610162,24.682419],[102.611327,24.683521],[102.612522,24.682998],[102.612543,24.681915],[102.614326,24.681673],[102.61405,24.680585],[102.615838,24.680072],[102.618801,24.680119],[102.623474,24.683713],[102.625834,24.684292],[102.626698,24.68566],[102.634703,24.685516],[102.634427,24.684428],[102.635648,24.682552],[102.636838,24.6823],[102.633906,24.680629],[102.634243,24.678468],[102.635448,24.677404],[102.635239,24.672797],[102.638528,24.671224],[102.6412,24.670995],[102.640914,24.670449],[102.646859,24.669459],[102.646573,24.668913],[102.649526,24.669501],[102.651911,24.668726],[102.652774,24.670093],[102.658981,24.671002],[102.660713,24.673467],[102.662485,24.673765],[102.66362,24.676491],[102.667145,24.67817],[102.668576,24.680901],[102.668561,24.681713],[102.667677,24.681428],[102.668847,24.682259],[102.668526,24.683608],[102.670288,24.684447],[102.675049,24.683438],[102.676194,24.685622],[102.677083,24.685636],[102.677349,24.687265],[102.680001,24.688118],[102.680553,24.690293],[102.683813,24.690344],[102.687595,24.694193],[102.686897,24.699869],[102.685074,24.702278],[102.68299,24.702788],[102.683256,24.704417],[102.680293,24.704371],[102.680278,24.705183],[102.677922,24.704334],[102.67761,24.705142],[102.677937,24.703522],[102.676169,24.702953],[102.673994,24.708335],[102.677846,24.708395],[102.680494,24.709519],[102.681348,24.711428],[102.680745,24.71196],[102.682804,24.712804],[102.682594,24.724175],[102.683453,24.725813],[102.685809,24.726661],[102.685201,24.727464],[102.686673,24.728029],[102.685417,24.731801],[102.688065,24.732925],[102.688623,24.734829],[102.690411,24.734315],[102.690683,24.735673],[102.689437,24.738903],[102.689372,24.742423],[102.687875,24.743212],[102.68994,24.743785],[102.690779,24.746506],[102.691668,24.74652],[102.691322,24.749223],[102.696025,24.751461],[102.698337,24.754747],[102.701558,24.756962],[102.702408,24.759142],[102.701454,24.762648],[102.702323,24.763744],[102.707917,24.765996],[102.709038,24.769534],[102.713762,24.770689],[102.712833,24.772842],[102.713708,24.773667],[102.71611,24.772079],[102.71608,24.773704],[102.717266,24.773722],[102.716693,24.77263],[102.71906,24.772937],[102.719628,24.774299],[102.721991,24.774877],[102.722302,24.774069],[102.724071,24.774638],[102.724679,24.773834],[102.725247,24.775197],[102.728509,24.775247],[102.729082,24.776339],[102.736224,24.775093],[102.73829,24.775666],[102.739451,24.777037],[102.739466,24.776225],[102.740346,24.77678],[102.739782,24.775147],[102.740672,24.77516],[102.741556,24.775444],[102.74089,24.779496],[102.74387,24.778729],[102.742397,24.778165],[102.743039,24.775466],[102.743607,24.776829],[102.744517,24.77576],[102.746286,24.776328],[102.746903,24.774983],[102.747759,24.776892],[102.747151,24.777695],[102.748337,24.777713],[102.747734,24.778245],[102.749227,24.777726],[102.750685,24.779102],[102.753349,24.779413],[102.754812,24.780518],[102.75417,24.783216],[102.755352,24.783505],[102.755614,24.785405],[102.754953,24.789186],[102.753747,24.790251],[102.754612,24.791618],[102.753951,24.795399],[102.755993,24.797326],[102.758361,24.797632],[102.759508,24.799816],[102.762153,24.801209],[102.762027,24.808248],[102.763461,24.810978],[102.768154,24.814027],[102.768382,24.817821],[102.767492,24.817808],[102.76777,24.818895],[102.76688,24.818882],[102.765654,24.82103],[102.766496,24.823751],[102.763466,24.827226],[102.767887,24.828917],[102.768446,24.830821],[102.769929,24.830843],[102.771311,24.836551],[102.768865,24.840576],[102.766155,24.842702],[102.768826,24.842742],[102.768223,24.843275],[102.769414,24.843022],[102.772051,24.844957],[102.772605,24.847131],[102.781658,24.85539],[102.781624,24.857285],[102.783706,24.857046],[102.785347,24.864923],[102.785327,24.866006],[102.78414,24.865989],[102.784715,24.86708],[102.783221,24.8676],[102.779865,24.872695],[102.777754,24.874559],[102.776868,24.874275],[102.775944,24.876157],[102.773239,24.878013],[102.773181,24.881261],[102.774066,24.881545],[102.773424,24.884244],[102.772534,24.884231],[102.772806,24.885589],[102.769464,24.889872],[102.771556,24.88909],[102.771828,24.890448],[102.768554,24.890941],[102.762194,24.897887],[102.759498,24.8992],[102.758267,24.901619],[102.752573,24.904512],[102.751045,24.906926],[102.750169,24.906101],[102.748932,24.90879],[102.747443,24.909038],[102.745623,24.911177],[102.744435,24.911159],[102.744406,24.912784],[102.740789,24.915708],[102.739241,24.919204],[102.738058,24.918916],[102.73175,24.922882],[102.723437,24.922755],[102.722522,24.924095],[102.721958,24.922462],[102.720186,24.921893],[102.718757,24.918892],[102.712885,24.915282],[102.711078,24.916608],[102.712528,24.918526],[102.711904,24.920141],[102.713042,24.922867],[102.712096,24.925831],[102.714411,24.929116],[102.71496,24.931562],[102.713129,24.934241],[102.709249,24.935265],[102.706033,24.932508],[102.706053,24.931425],[102.704534,24.933297],[102.704861,24.931677],[102.702516,24.930016],[102.703446,24.927864],[102.697333,24.921271],[102.694067,24.92122],[102.691129,24.91955],[102.69131,24.925781],[102.68978,24.928195],[102.690021,24.931177],[102.688189,24.933856],[102.689367,24.934416],[102.689347,24.935499],[102.688174,24.934669],[102.687258,24.936008],[102.686388,24.934912],[102.683997,24.935687],[102.683122,24.934861],[102.683444,24.933512],[102.681688,24.932131],[102.680511,24.931571],[102.679011,24.93236],[102.680333,24.941046],[102.681792,24.942422],[102.683759,24.94841],[102.686084,24.951154],[102.685107,24.955743],[102.681785,24.95867],[102.674295,24.962073],[102.665371,24.962746],[102.657082,24.961262],[102.656473,24.962064],[102.652315,24.961999],[102.649325,24.963035],[102.649796,24.969541],[102.648257,24.98793],[102.650004,24.989853],[102.647914,24.990362],[102.649399,24.990385],[102.649926,24.993914],[102.649025,24.994441],[102.649306,24.995258],[102.646618,24.996028],[102.6463,24.997106],[102.643627,24.997064],[102.64442,25.002221],[102.651727,25.008565],[102.655256,25.010516],[102.65645,25.010264],[102.658201,25.011916],[102.659687,25.01194],[102.659072,25.013013],[102.660834,25.014124],[102.661981,25.016308],[102.66759,25.018292],[102.668425,25.021284],[102.670807,25.02105],[102.671089,25.021867],[102.671401,25.02106],[102.671038,25.024574]];
var TARGET_GEOM = ee.Geometry.Polygon([TARGET_OUTLINE], null, false).buffer(120);
// Keep the 2019 JRC component containing the fixed lake-center seed.
// The polygon was drawn from that component before any model inference.
var water = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filter(ee.Filter.eq('year', 2019)).first())
  .select('waterClass').gte(2).updateMask(ee.Image.constant(1).clip(TARGET_GEOM)).unmask(0, false);
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
