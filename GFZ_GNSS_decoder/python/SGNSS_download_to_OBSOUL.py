# %%
import datetime as dt
import gzip
import os
import sys
from posix import environ

import numpy as np
import pandas as pd
# %%
import mlat

sgnss_file_type = { 
    'GFZ' : { 
        'ftp' : 'ftp.gfz-potsdam.de',
        'slants_ULTRA_RAPID_EPOS8' : '/pub/home/GNSS/products/nrttrop/slants_ULTRA_RAPID_EPOS8/y%Y/slant_h_o_%Y%m%d%H00_%Y%m%d%H59_mult_gf1u.dat.gz',
        'slants_RAPID_EPOS8' : '/pub/home/GNSS/products/nrttrop/slants_RAPID_EPOS8/y%Y/slant_h_o_%Y%m%d%H00_%Y%m%d%H59_mult_gf1r.dat.gz' ,
        'slants_GLOBAL_EPOS8' : '/pub/home/GNSS/products/nrttrop/slants_GLOBAL_EPOS8/y%Y/slant_h_o_%Y%m%d%H00_%Y%m%d%H59_mult_gf1r.dat.gz' ,
        'REPRO' : '/pub/home/GNSS/products/nrttrop/REPRO/slants_EPOS8/y%Y/slv_slant_repro_%Y_%j_23.gz' 
    }
}

def create_spatial_thinning(df,site_sep=50):
    """
    site_sep: minimal separation distance in km
    """
    df=df.reset_index()
    N=len(df)
    site_del=[]
    j=range(N)
    print(f"number of sites before thinning {len(df)}")
    for i,x in df.iterrows():
        if i in site_del:
            continue
        d=lonlat2aeqd_dist(x.lon,x.lat,df.lon.values,df.lat.values)
        d[i]=9e9
        site_del+=[ i for i in j if d[i]<site_sep and i not in site_del ]
    sites=[ i for i in j if i not in site_del]
    df=df.iloc[sites]
    print(f"number of sites after thinning  {len(df)} (separation {site_sep})")
    return df

def lonlat2aeqd_dist(lon0,lat0,lon_ref0,lat_ref0,outType='da'):
    radius=6378.
    DEG2RAD=np.deg2rad(1)
    lon=lon0*DEG2RAD;
    lat=lat0*DEG2RAD;
    lon_ref=lon_ref0*DEG2RAD;
    lat_ref=lat_ref0*DEG2RAD;
    L_Lref=-(lon-lon_ref);
  
    #/*Calculation of azimuth and highest point of great circle*/
    azim=np.arctan2(-np.sin(L_Lref),np.cos(lat_ref)*np.tan(lat)-np.sin(lat_ref)*np.cos(L_Lref));
    latstar=np.arccos(np.sin(azim)*np.cos(lat_ref));
  
    #/*First estimate of arc lenghts from the reference position and the target to
    #  highest point of great circle.*/
    Sref=np.arctan2(np.cos(azim),np.tan(lat_ref));
    Lref=np.arctan2(np.tan(Sref),np.cos(latstar));
    S=np.arctan(np.tan(L_Lref+Lref)*np.cos(latstar));
    dist=abs(radius*(Sref-S));
    return dist

def printOBSOUL(df,andtg,outfile="SGNSS.OBSOUL"):
    f=open(outfile,'w')
    # skipping header: printed in Bator f.write("%s\n"%(andtg.strftime("%Y%m%d %H")))
    for i,x in df.iterrows():
        f.write("17 19 111 %f %f ' %s_%02d' %s %f 1 11111 0\n"%(x.lat,x.lon,x.site,x.sat,x.dtg.strftime("%Y%m%d %H%M%S"),x.hMSL))
        f.write("129 %f %f %f %f\n"%(x.az,x.el,x.spd,x.sat+0.01/np.sin(x.el/180*3.14159)))
    f.close()
    print(" OBSSOUL-file : {}".format(outfile))
    print("  number of obs   : {}".format(len(df)))
    print("  number of sites : {}".format(len(set(df.site))))
    print("  number of sats  : {}".format(len(set(df.sat))))

    for p in  'lon','lat','dtg','hMSL','az','el':
        print("  range {}  : {} to {}".format(p,df[p].min(),df[p].max()))

def download_ftp(dtg=None,type_of_file=['GFZ','slants_ULTRA_RAPID_EPOS8'],dhours=[-2,-1]):
    """
    type can be :
    slants_GLOBAL_EPOS8
    slants_RAPID_EPOS8
    slants_ULTRA_RAPID_EPOS8
    REPRO:REPRO/slants_EPOS8/y2021
    """
    if dtg == None:
        dtg = dt.datetime.utcnow() 
    files=[]
    if 'GFZ' == type_of_file[0] and 'REPRO' in type_of_file[1]:
        file=dtg.strftime('slant%j.gz')
        ftp_server=sgnss_file_type[type_of_file[0]]['ftp']
        if not os.path.isfile(file):
            ftp_file=dtg.strftime(sgnss_file_type['GFZ'][type_of_file[1]])
            cmd="echo 'get {0} {1}'| ftp {2}".format(ftp_file,file,ftp_server)
            os.system(cmd)
        else:
            files=[file]
    else:
        ftp_server=sgnss_file_type[type_of_file[0]]['ftp']
        for dh in dhours:
            dtg0=dtg + dt.timedelta(hours=dh)
            file=dtg0.strftime(sgnss_file_type[type_of_file[0]][type_of_file[1]])
            cmd="rm -f slant{1}.gz;echo 'get {0} slant{1}.gz'| ftp {2}".format(file,dh,ftp_server)
            os.system(cmd)
            if os.path.isfile("slant{}.gz".format(dh)):
                files.append("slant{}.gz".format(dh))
            else:
                print(" data for {} is missing".format(dtg0))

    return files

def read_slant(file,dtg=None,sites_only=False):
    if not os.path.isfile(file):
        print(f"file {file} not found")
        return []
    type=0
    idx=[]
    nam=[]
    llh=[]
    hc=[]
    lat=np.zeros([1000])
    lon=np.zeros([1000])
    try:
        minlat=float(os.environ["MINLAT"])
        maxlat=float(os.environ["MAXLAT"])
        minlon=float(os.environ["MINLON"])
        maxlon=float(os.environ["MAXLON"])
    except:
        minlat,minlon,maxlat,maxlon=-90,-180,90,360
    verb=0
    hg=[0.,0.]
    i=3
    res=[]
    contents={}
    n=0
    print(f"reading {file} : datasets")
    with gzip.open(file,'r') as fin:
        for line in fin:
            line=line.decode("utf-8")
            if line[0]=='+':
                cont=line[1:-1]
                contents[cont]={}
                contents[cont]['start']=n+1
                if cont == 'slants':
                    contents[cont]['end']=-999
                    break
            if line[0]=='-':
                contents[cont]['end']=n-2
            n+=1
    start=contents['slant_delay_param']['start']
    end=contents['slant_delay_param']['end']
    sizes=pd.read_csv(file,dtype={'name':'string','value':'string'},usecols=[0,1],delim_whitespace=True,skiprows=start,nrows=end-start,index_col=0).T
    sizes=sizes.to_dict(orient='records')[0]

    number_of_stations=int(sizes['number_of_stations'])
    number_of_satellites=int(sizes['number_of_satellites'])
    number_of_slants=int(sizes['number_of_slants'])
    mjd0=float(sizes['reference_MJD'])
    dtg_mdj=dt.datetime(1858,11,17) + dt.timedelta(days=mjd0) 
    start=contents['sta_number_name_coordinate']['start']
    xyz=pd.read_csv(file,skiprows=start,nrows=number_of_stations,delim_whitespace=True,names=['num','site','x','y','z'],\
                    dtype={'num':'int','site':'string','x':'float','y':'float','z':'float'},comment='*',error_bad_lines=False)

    xyz['lat'],xyz['lon'],xyz['h']=mlat.ecef2llh((xyz['x'].values,xyz['y'].values,xyz['z'].values) )
    if not sites_only:
        print( f"number of sites {len(xyz)} (total)")
    xyz=xyz[(minlon<xyz.lon)&(xyz.lon<maxlon)&(minlat<xyz.lat)&(xyz.lat<maxlat)]
    if not sites_only:
        print( f"number of sites {len(xyz)} (domain)")

    if sites_only:
        return xyz

    start=contents['slants']['start']
    slants=pd.read_csv(file,skiprows=start,nrows=number_of_slants,delim_whitespace=True,\
                        names=['d_mjd','num','sat','el','az', 'spd','szd'],\
                        dtype={'d_mjd':'float','num':'int','sat':'int','el':'float','az':'float', 'spd':'float','szd':'float'},\
                        comment='*',error_bad_lines=False)
    xyz['hMSL']=mlat.wgs84_height(xyz.lat, xyz.lon)
    slants['dtg'] = [ dtg_mdj + dt.timedelta(days=dmjd) for dmjd in slants.d_mjd.values]

    res=slants.merge(xyz,on = 'num', how='left')
    res=res[['sat','lat','lon','site','dtg','hMSL','az','el','spd']]

    return res

def plotSites(df,region="europe"):
    import matplotlib.pyplot as plt

    # Third-party modules
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
    fig = plt.figure(figsize=[12,8])
    if region == "europe":
        minLon,maxLon=-20,30
        minLat,maxLat=35,70
    else:
        minLon,maxLon=-180,180
        minLat,maxLat=-90,90
         
    # setup mercator map projection.
    if maxLon-minLon<90:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(central_longitude = np.mean([minLon, maxLon])))
        ax.set_extent([minLon, maxLon, minLat, maxLat], crs=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Miller())

    # Resolution of all map features
    resol = '50m'  # use data at this scale
    # Add Feature/Tiles
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', 
                                        scale=resol, edgecolor='k', alpha=0.5, linestyle='-', facecolor='none')
    land = cfeature.NaturalEarthFeature('physical', 'land', \
        scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', \
        scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    lakes = cfeature.NaturalEarthFeature('physical', 'lakes', \
        scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
    rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', \
        scale=resol, edgecolor='b', facecolor='none')
    # ax.add_feature(borders, zorder=7)
    # ax.add_feature(land, zorder=0)
    ax.add_feature(ocean, alpha=0.5, zorder=0)
    # ax.add_feature(lakes, alpha=0.15, zorder=1)
    # ax.add_feature(rivers, alpha=0.15, zorder=1)

    # Coastlines
    # ax.coastlines(resolution='50m', zorder=10)
    df=df[(df.lon>minLon)&(df.lon<maxLon)&(df.lat>minLat)&(df.lat<maxLat+20)]
    ax.set_title(f"sites processed by GFZ in near real time\n{df.dtg.min().replace(microsecond=0)} - {df.dtg.max().replace(microsecond=0)}\nnumber of sites: {len(set(df.site))}  number of obs: {len(df)}")
    cb = ax.plot(df['lon'].tolist(), df['lat'].tolist(), '^k', transform=ccrs.PlateCarree(),zorder=10)
    plt.tight_layout()
    fig.savefig(f"locations_{region}.png")

if __name__ == "__main__": 
    n=len(sys.argv)
    argv=sys.argv
    i=1
    plot=False
    andtg=None
    files=None
    thinning=False
    outfile="SGNSS.OBSOUL"
    while i<n:
        try:
            if argv[i]=='-f':
                files=argv[i+1].split(',')
            if argv[i]=='-o':
                outfile=argv[i+1]
            if argv[i]=='-d':
                andtg=dt.datetime.strptime(sys.argv[i+1],"%Y%m%d%H")
            if argv[i]=='-T':
                type_of_file=argv[i+1].split(',')
                if type_of_file[0] not in sgnss_file_type:
                    print(f"source definition not found in {sgnss_file_type}")
                    quit()
                if type_of_file[1] not in sgnss_file_type[type_of_file[0]]:
                    print(f"file definition not found in {sgnss_file_type[type_of_file[0]]}")
                    quit()

            if argv[i]=='-p':
                plot=True
            if argv[i]=='-t':
                thin_file=argv[i+1]
                if os.path.isfile(thin_file):
                    df_thin=pd.read_csv(sys.argv[i+1])
                    print(f"thinning applied from file {thin_file}")
                else:
                    df_thin=None
                    print(f"thinning applied and written to {thin_file}")
                thinning=True
        except:
            print(f"argument error : {argv[i]}")
            quit()
        i+=1
    try:
        window = int(os.environ["SGNSS_window_length"]) # time window in seconds
        step = int(os.environ["SGNSS_window_step"]) # time window in seconds
        Nstep = int(os.environ["SGNSS_window_num_of_step"]) # time window in seconds
    except:
        print("using default settings")
        window = 900
        step = 20*60
        Nstep = 3
    try:
        minlat=float(os.environ["MINLAT"])
        maxlat=float(os.environ["MAXLAT"])
        minlon=float(os.environ["MINLON"])
        maxlon=float(os.environ["MAXLON"])
    except:
        if not plot:
            os.environ["MINLAT"] = "45"
            os.environ["MAXLAT"] = "65"
            os.environ["MINLON"] = "-15"
            os.environ["MAXLON"] = "25"

    if files is None and andtg is not None:
        files = download_ftp(type_of_file=type_of_file,  dtg=andtg,dhours=[-1,0])
        if len(files)==0:
            print(f"no GFZ slant data found")
            quit()
    if thinning and df_thin is None:
        df= read_slant(files[0],sites_only=True)
        df_thin=create_spatial_thinning(df)
        df_thin.to_csv(thin_file,index=False)
        print("thin file written")
    df=[]
    for file in files:
        df.append( read_slant(file=file) )
    if len(df)==0:
        print(f"no GFZ slant data found")
        quit()
    df=pd.concat(df)
    if thinning:
        print(f"number of sites before thinning {len(df)}")
        df=df[df.site.isin(set(df_thin.site))]
        print(f"number of sites after  thinning {len(df)}")
    if andtg is None:
        dtgbeg = df.dtg.min().replace(minute=0,second=0,microsecond=0)
        dtgend = df.dtg.max() + dt.timedelta(seconds=window/2)
        andtg = dtgbeg + dt.timedelta(hours=1)
    else:
        dtgbeg = andtg - dt.timedelta(seconds = Nstep*step)
        dtgend = andtg + dt.timedelta(seconds = Nstep*step)
    dfout=[]
    dtg = dtgbeg
    while dtg <= dtgend:
        df0=df[abs((df.dtg - dtg)/dt.timedelta(seconds=1))<window/2]
        # select observations with lowest elevation
        df0=df0.loc[df0.groupby(['site','sat']).el.idxmin()]
        df0['refdtg'] = dtg
        print(f" number of obs in window {dtg} : {len(df0)}")
        dfout.append(df0)
        dtg += dt.timedelta(seconds=step)
        if plot:
            break
    dfout = pd.concat( dfout )
    printOBSOUL(dfout,andtg,outfile=outfile)
    if plot:
        plotSites(dfout,"global")
        plotSites(dfout)
    quit()
