import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';

function useForceUpdate(){
  const [value, setValue] = useState(0); // integer state
  return () => setValue(value => value + 1); // update the state to force render
}

const SampleChoice: React.FC = () => {
  const forceUpdate = useForceUpdate();
  // const [episodeNumber, setEpisodeNumber] = useState<number>(1);
  const nextEpisodeAvailable = useRef<boolean>(false);
  const episodeNumber = useRef<number>(1);
  const submitChoice = (choice: number) => {
    axios.post(`/human_choice/${choice}`)
      .then(res => {
        nextEpisodeAvailable.current = false;
        episodeNumber.current = res.data.displayed_episode;
        forceUpdate();
      });
  };

  useEffect(()=>{
    axios.get(`/displayed_episode`)
      .then(res => {
        episodeNumber.current = res.data.displayed_episode;
        forceUpdate();
      });
  },[]);

  const checkNewEpisode = () => {
    axios.get(`/videos/video_two_${episodeNumber.current}.mp4`)
      .then(() => {
        nextEpisodeAvailable.current = true;
        forceUpdate();
      })
      .catch(() => {
        console.log('not yet');
      });
  };
  const buttonStyle = "bg-indigo-500 hover:bg-blue-500 p-2 active:bg-cyan-500 rounded-xl";
  return (
    <>
      {nextEpisodeAvailable.current ? 
        <>
          <div className="flex flex-col mx-auto mt-28">
            <div className="flex flex-row mx-auto mb-10">
                Episode {episodeNumber.current}
            </div>
            <div className="flex flex-row">
              <div className="flex flex-col mx-4">
                <video className="rounded-xl" autoPlay loop muted width="320" height="320" src={`videos/video_one_${episodeNumber.current}.mp4`}/>
                <button className={`${buttonStyle} mt-2`} onClick={()=>(submitChoice(1))}>1</button>
                <button className={`${buttonStyle} mt-2`} onClick={()=>(submitChoice(3))}>Equally good</button>
              </div>
              <div className="flex flex-col mx-4">
                <video className="rounded-xl" autoPlay loop muted width="320" height="320" src={`videos/video_two_${episodeNumber.current}.mp4`}/>
                <button className={`${buttonStyle} mt-2`} onClick={()=>(submitChoice(2))}>2</button>
                <button className={`${buttonStyle} mt-2`} onClick={()=>(submitChoice(4))}>Discard</button>
              </div>
            </div>
          </div>
        </>
      :
        <div className="flex justify-items-center w-screen h-screen">
          <button onClick={checkNewEpisode} className={`${buttonStyle} m-auto`}>
            Go to the next episode
          </button>
        </div>
      }
    </>
  );
};

export default SampleChoice;
