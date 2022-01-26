import axios from 'axios';
import React, { useEffect, useState, useRef } from 'react';
import VideosContainer from "./VideosWrapper"

function useForceUpdate(){
  const [value, setValue] = useState(0); // integer state
  return () => setValue(value => value + 1); // update the state to force render
}

const History: React.FC = () => {
  const episodeNumber = useRef<number>(0);
  const forceUpdate = useForceUpdate();
  const [videos, setVideos] = useState<JSX.Element[]>([<></>])
  var videoList = [<></>];
  useEffect(()=>{
    console.log('mount');
    axios.get(`/history_size`)
      .then((res) => {
        console.log(res.data);
        episodeNumber.current = res.data.records;
        setVideos(episodeNumber.current > 1 ? [...Array(episodeNumber.current)].map((x, i) =>
          <VideosContainer idx={i+1}/>
        ) : [<></>]);
        forceUpdate();
      });
  },[]);


  document.querySelectorAll("video").forEach(v=>(v.playbackRate=0.7));
  return (
    <div className="mx-auto mt-10">
      <div className="flex flex-col overflow-auto justify-items-center align-center h-4/5 px-6">
        {videos}
      </div>
    </div>
  );
};

export default History;
